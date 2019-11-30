from .OneShotSkeleton import OneShotSkeleton as OSS
from mxnet import nd
import cv2
import numpy as np


def video_samples(cap, interval) :
    '''
    비디오 샘플을 interval 간격으로 추출하되
    이 간격에 0~2 만큼의 방해를 적용합니다.
    '''
    n = 0
    while True :
        ret, frame = cap.read()
        if not ret :
            break
        else :
            bias = np.random.randint(low=0, high=3)
            if (n + bias) % interval == 0 :
                    yield frame
        n += 1
    
class VideoSkeleton() :
    
    def __init__(self, ctx) :
        self.ctx = ctx
        self.oss = OSS(ctx=ctx)
        
    def predict(self, path, interval, bbox_thr, augument=False) :
        
        coords, confidences = [], []
        aug_coords = []
        cap = cv2.VideoCapture(path)
        width = cap.get(3)
        for frame in video_samples(cap, interval) :
            frame = nd.array(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)).astype('uint8')
            pred_coords, confidence, bbox = self.oss(frame, bbox_thr=bbox_thr)
            coords.append(pred_coords[0])
            confidences.append(confidence[0])
            
            if augument :
                aug_coord = pred_coords[0].copyto(self.ctx)
                if len(bbox) > 0 :
                    aug_coord[:,0] = width - aug_coord[:,0]
                aug_coords.append(aug_coord)
                
        cap.release()
        
        coords = nd.stack(*coords)
        confidences = nd.stack(*confidences)
        if augument :
            aug_coords = nd.stack(*aug_coords)
            return nd.stack(coords, aug_coords), nd.stack(confidences, confidences)
        else :
            return coords, confidences