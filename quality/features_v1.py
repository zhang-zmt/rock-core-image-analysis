"""只读岩芯裁片的可解释指标、监督阈值拟合和近邻一致性检查。

只依赖NumPy、Pillow。输出三档是决策等级，不是校准概率；不修改任何图像。
人工原因未勾选的项目不当作阴性标签；本版只拟合整体合格/不合格标签。
"""
# 模块补充说明（仅注释，不改变行为）：
#   - 输入统一为 250×350 单帧图像；全部特征为手工设计的可解释指标；
#   - 模型 = 单侧阈值规则 + 特征空间 kNN，输出 keep / exclude / review 三档决策等级（非校准概率）；
#   - 所有函数只读文件，不写回图片或标签。
from __future__ import annotations
from pathlib import Path
import hashlib
import math
import numpy as np
from PIL import Image, ImageFilter

# 引擎版本号，会写入冻结模型，用于追溯模型与代码的对应关系
VERSION='1.0.0'
# 特征顺序即模型向量顺序：拟合、预测、序列化都必须严格按此顺序取值，错位会导致特征含义错配
FEATURE_NAMES=['edge_black','black_fraction','red_blob','crack_ridge','white_clip',
               'bright_clip','blur_badness','chroma_ratio','stripe_strength','gray_std','anisotropy']
# 参与“单侧阈值规则”的特征及其对应的可读原因；其余特征只用于 kNN 距离度量
RULE_FEATURES={'edge_black':'疑似外部背景/边缘','red_blob':'疑似人工记号',
               'crack_ridge':'疑似黑色裂缝','white_clip':'疑似过曝','blur_badness':'纹理清晰度不足'}


# 计算文件的 SHA-256 摘要，用于对人工标签等做“内容是否被改动”的封存校验。
# 以 1MB 分块流式读取，避免把大文件整体载入内存。
def sha256_file(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as f:
        for b in iter(lambda:f.read(1024*1024),b''):h.update(b)
    return h.hexdigest()


# 连通域分析（不依赖 OpenCV/scipy 的实现）：
#   1) 逐行把连续的 1 段压缩为一个“游程”，记录 (y, 起点, 终点)；
#   2) 相邻两行游程若列区间重叠，就用并查集合并为同一连通块；
#   3) 按并查集根汇总面积、外接框，以及是否触及图像边界。
# 返回值为列表，每项格式：[面积, x1, y1, x2, y2, 是否接边]
def components(mask):
    """4连通，逐行游程并查集；返回面积、边界框及是否接边。"""
    h,w=mask.shape;parents=[];runs=[];previous=[]
    # 并查集查找，带路径压缩：把节点直接挂到祖先，降低后续查找深度
    def root(i):
        while parents[i]!=i:
            parents[i]=parents[parents[i]];i=parents[i]
        return i
    for y in range(h):
        # 行首尾各补 0：0→1 的跳变记为 +1（游程起点），1→0 记为 -1（游程终点）
        diff=np.diff(np.pad(mask[y].astype(np.int8),(1,1)))
        starts=np.flatnonzero(diff==1);ends=np.flatnonzero(diff==-1)
        current=[];p=0
        for a,b in zip(starts.tolist(),ends.tolist()):
            i=len(parents);parents.append(i);runs.append((y,a,b));current.append((a,b,i))
            # 跳过上一行中已结束（右端 <= 当前起点）的游程，减少无谓比较
            while p<len(previous) and previous[p][1]<=a:p+=1
            q=p
            # 与上一行所有“列区间有交集”的游程（上一行起点 < 当前右端）合并为同一连通块
            while q<len(previous) and previous[q][0]<b:
                j=root(previous[q][2]);r=root(i)
                if j!=r:parents[r]=j
                q+=1
        previous=current
    # 按并查集根节点汇总：面积累加，外接框取坐标的极小/极大值
    result={}
    for i,(y,a,b) in enumerate(runs):
        r=root(i)
        # 首次遇到该根时初始化累加器：[面积, x1, y1, x2, y2, 是否接边]
        if r not in result:result[r]=[0,a,y,b,y+1,False]
        v=result[r];v[0]+=b-a;v[1]=min(v[1],a);v[2]=min(v[2],y)
        v[3]=max(v[3],b);v[4]=max(v[4],y+1)
        # 触及左/右/上/下任一边界即认为该块接到图像边缘（用于识别“外部背景/黑边”）
        v[5]=bool(v[5] or a==0 or b==w or y==0 or y==h-1)
    return list(result.values())


# 二值形态学（用 PIL 的 MinFilter/MaxFilter 模拟腐蚀/膨胀实现）：
#   operation='open' —— 先腐蚀后膨胀（开运算，去小噪点）；
#   其他取值一律按闭运算处理 —— 先膨胀后腐蚀（补小孔）。
# 掩码以 0/255 灰度图承载，滤波后再用 >0 还原为布尔掩码。
def morphology(mask,operation,size):
    with Image.fromarray(mask.astype(np.uint8)*255) as im:
        if operation=='open':out=im.filter(ImageFilter.MinFilter(size)).filter(ImageFilter.MaxFilter(size))
        else:out=im.filter(ImageFilter.MaxFilter(size)).filter(ImageFilter.MinFilter(size))
        return np.asarray(out)>0


# 从单张岩芯裁片提取 11 维手工特征。
# 先校验输入合法性（必须 250×350、单帧），再统一转 RGB 计算；全程只读，不写回图像。
def extract_features(path):
    with Image.open(path) as im:
        # 尺寸或帧数不符合约定直接报错，避免在异常输入上算出误导性特征值
        if im.size!=(250,350) or getattr(im,'n_frames',1)!=1:
            raise ValueError(f'需要250×350单帧图像：{path}')
        rgb=np.asarray(im.convert('RGB'),dtype=np.float64)
    h,w=rgb.shape[:2];area=h*w
    # 灰度按 ITU-R 601 亮度系数加权；v 为三通道最大值（用于高光/过曝判断）
    gray=rgb@np.array([.299,.587,.114]);v=rgb.max(axis=2)
    # “硬黑”像素：三通道最大值 <=12，近似纯黑区域
    hard_black=v<=12
    comps=components(hard_black)
    # 边缘黑色占比：只累加接到图像边界的黑色连通块面积，用于识别外部背景/黑边
    edge_black=sum(c[0] for c in comps if c[5])/area
    r,g,b=rgb.transpose(2,0,1)
    # 偏红像素：红通道显著高于绿/蓝，且红-最小通道 >45，近似人工红色记号
    red=(r>70)&(r>1.35*g)&(r>1.2*b)&((r-np.minimum(g,b))>45)
    # 开运算去掉细小噪点，再取最大红色连通块面积占比（最大块通常就是记号笔主体）
    red=morphology(red,'open',3)
    red_blob=max((c[0] for c in components(red)),default=0)/area
    with Image.fromarray(gray.clip(0,255).astype(np.uint8)) as gim:
        # closed 用 9×9 闭运算估计局部背景亮度；smooth 用 0.7 高斯做轻微去噪以稳定梯度
        closed=np.asarray(gim.filter(ImageFilter.MaxFilter(9)).filter(ImageFilter.MinFilter(9)),dtype=float)
        smooth=np.asarray(gim.filter(ImageFilter.GaussianBlur(.7)),dtype=float)
    # 裂纹候选像素：比局部背景暗 >=16 且自身较暗（<55），即“脊线/裂缝”形态
    ridge=((closed-gray)>=16)&(gray<55)
    crack=0
    for c in components(ridge):
        # 裂缝判别：外接框长边 >=50 像素、足够细长（面积/长度 <=12）、且总面积 >=20 才计入
        length=max(c[3]-c[1],c[4]-c[2])
        if length>=50 and 0<c[0]/length<=12 and c[0]>=20:crack=max(crack,c[0])
    # 有效区域：排除硬黑背景与已判为人工红记号的像素
    valid=(~hard_black)&(~red)
    valid[0:3,:]=False;valid[-3:,:]=False;valid[:,0:3]=False;valid[:,-3:]=False
    # 中心差分近似梯度（x、y 方向），边缘列/行留 0
    gx=np.zeros_like(gray);gy=np.zeros_like(gray)
    gx[:,1:-1]=(smooth[:,2:]-smooth[:,:-2])/2
    gy[1:-1]=(smooth[2:]-smooth[:-2])/2
    energy=gx*gx+gy*gy;block=[]
    for y in range(0,h,50):
        for x in range(0,w,50):
            mask=valid[y:y+50,x:x+50]
            if mask.mean()>=.9:block.append(float(np.median(energy[y:y+50,x:x+50][mask])))
    texture=float(np.median(block)) if block else float(np.median(energy[valid])) if valid.any() else 0.
    # 高频色差和周期峰只作为邻域特征；不直接视为确定失真。
    chroma=r-g
    ch=np.mean(np.abs(np.diff(chroma,axis=0)))+np.mean(np.abs(np.diff(chroma,axis=1)))
    lum=np.mean(np.abs(np.diff(gray,axis=0)))+np.mean(np.abs(np.diff(gray,axis=1)))
    profile=np.mean(chroma,axis=0);profile-=profile.mean()
    power=np.abs(np.fft.rfft(profile))**2;power[:3]=0
    stripe=float(power.max()/(power.sum()+1e-9))
    vx=float(np.mean(gx[valid]**2)) if valid.any() else 0.
    vy=float(np.mean(gy[valid]**2)) if valid.any() else 0.
    return {'edge_black':edge_black,'black_fraction':float(hard_black.mean()),
            'red_blob':red_blob,'crack_ridge':crack/area,
            'white_clip':float((rgb.min(axis=2)>=245).mean()),
            'bright_clip':float((v>=250).mean()),'blur_badness':-math.log1p(texture),
            'chroma_ratio':float(ch/(lum+1)),'stripe_strength':stripe,
            'gray_std':float(np.std(gray[valid])) if valid.any() else 0.,
            'anisotropy':abs(vx-vy)/(vx+vy+1e-9)}


def vector(features):return np.array([features[n] for n in FEATURE_NAMES],dtype=float)


def fit_model(items):
    if len(items)<10:raise ValueError('校准样本不足10张。')
    if len({r['group_id'] for r in items})!=len(items):
        raise ValueError('本版校准要求每张来自不同原图，防止原图重复加权。')
    x=np.stack([vector(r['features']) for r in items]);y=np.array([r['bad'] for r in items])
    if set(y.tolist())!={0,1}:raise ValueError('校准必须同时有合格与不合格样本。')
    center=np.median(x,axis=0);q25,q75=np.percentile(x,[25,75],axis=0)
    scale=np.maximum(q75-q25,np.maximum(np.std(x,axis=0)*.25,1e-6))
    z=np.clip((x-center)/scale,-8,8)
    distances=np.sqrt(np.mean((z[:,None,:]-z[None,:,:])**2,axis=2))
    np.fill_diagonal(distances,np.inf)
    radius=float(np.quantile(np.partition(distances,4,axis=1)[:,4],.9))
    rules=[]
    for feature,reason in RULE_FEATURES.items():
        values=x[:,FEATURE_NAMES.index(feature)];unique=np.unique(values)
        thresholds=(unique[:-1]+unique[1:])/2
        best=None
        for threshold in thresholds:
            hit=values>=threshold
            fp=int(np.sum(hit&(y==0)));tp=int(np.sum(hit&(y==1)))
            # 仅拟合没有误排已知合格图、且至少有3张排除图支持的单侧条件。
            if fp==0 and tp>=3:
                candidate=(tp,float(threshold))
                if best is None or candidate>best:best=candidate
        if best:
            rules.append({'feature':feature,'op':'>=','threshold':best[1],
                          'development_support_bad':best[0],'development_false_exclusions':0,'reason':reason})
    return {'engine_version':VERSION,'method':'supervised_threshold_rules_with_handcrafted_feature_knn',
            'feature_names':FEATURE_NAMES,'center':center.tolist(),'scale':scale.tolist(),
            'reference_features':x.tolist(),'reference_labels':y.tolist(),
            'reference_sample_ids':[r['sample_id'] for r in items],
            'reference_group_ids':[r['group_id'] for r in items],
            'k':5,'max_neighbor_distance':radius,'rules':rules,
            'keep_votes_required':5,'exclude_votes_required':4,
            'confidence_semantics':'decision_tier_not_calibrated_probability',
            'neural_network_training':False,'multilabel_reason_training':False}


def predict(model,features):
    x=vector(features);refs=np.array(model['reference_features'])
    center=np.array(model['center']);scale=np.array(model['scale'])
    z=np.clip((x-center)/scale,-8,8);rz=np.clip((refs-center)/scale,-8,8)
    dist=np.sqrt(np.mean((rz-z)**2,axis=1));order=np.argsort(dist,kind='stable')[:model['k']]
    bad_votes=int(np.array(model['reference_labels'])[order].sum());hits=[]
    for r in model['rules']:
        if features[r['feature']]>=r['threshold']:hits.append(r['reason'])
    in_range=bool(dist[order[-1]]<=model['max_neighbor_distance'])
    if hits and bad_votes>=model['exclude_votes_required'] and in_range:decision='likely_exclude'
    elif not hits and bad_votes==0 and in_range:decision='likely_keep'
    else:decision='review'
    return {'machine_decision':decision,'rule_triggers':hits,'neighbor_bad_votes':bad_votes,
            'neighbor_count':len(order),'neighbors':[model['reference_sample_ids'][i] for i in order],
            'fifth_neighbor_distance':float(dist[order[-1]]),'within_development_range':in_range,
            'confidence_semantics':model['confidence_semantics']}


def metrics(pairs):
    n=len(pairs);auto=[(y,p) for y,p in pairs if p!='review'];good=sum(y==0 for y,p in pairs);bad=n-good
    false_exclude=sum(y==0 and p=='likely_exclude' for y,p in pairs)
    missed_bad=sum(y==1 and p=='likely_keep' for y,p in pairs)
    return {'total':n,'automatic_count':len(auto),'review_count':n-len(auto),
            'automatic_coverage':len(auto)/n if n else None,
            'selective_accuracy':sum((p=='likely_exclude')==bool(y) for y,p in auto)/len(auto) if auto else None,
            'false_exclusion_count':false_exclude,'false_exclusion_rate_among_all_good':false_exclude/good if good else None,
            'missed_bad_count':missed_bad,'missed_bad_rate_among_all_bad':missed_bad/bad if bad else None}


def grouped_oof(items):
    # 首轮每图一组；按标签分层并用固定SHA排序分5折，所有变换在训练折内拟合。
    folds={}
    for label in [0,1]:
        subset=sorted((r for r in items if r['bad']==label),
                      key=lambda r:hashlib.sha256(('oof:'+r['group_id']).encode()).hexdigest())
        for i,r in enumerate(subset):folds[r['group_id']]=i%5
    pairs=[];rows=[]
    for fold in range(5):
        train=[r for r in items if folds[r['group_id']]!=fold]
        test=[r for r in items if folds[r['group_id']]==fold]
        assert not ({r['group_id'] for r in train}&{r['group_id'] for r in test})
        model=fit_model(train)
        for r in test:
            p=predict(model,r['features']);pairs.append((r['bad'],p['machine_decision']))
            rows.append({'sample_id':r['sample_id'],'fold':fold,'human_bad':r['bad'],**p})
    return {'purpose':'development_grouped_oof_not_final_acceptance','folds':5,'metrics':metrics(pairs),'rows':rows}
