import gluoncv as gcv
from gluoncv import model_zoo, data, utils
from gluoncv.data.transforms.pose import detector_to_simple_pose, heatmap_to_coord
from mxnet import nd

class OneShotSkeleton() :
    '''
    바운더리 박스 디텍터 : ssd_512_mobilenet1.0
    포즈 프레딕터 : simple_pose_resnet18_v1b
    더 좋은 성능의 디텍터와 프레딕터로 변경될 수 있음.
    '''
    def __init__(self, ctx) :
        self.ctx = ctx
        self.detector = model_zoo.get_model('ssd_512_mobilenet1.0_coco', pretrained=True, ctx=ctx)
        self.detector.reset_class(["person"], reuse_weights=['person'])
        self.pose_net = model_zoo.get_model('simple_pose_resnet18_v1b', pretrained='ccd24037', ctx=ctx)
    
    def predict(self, frame, bbox_thr=0.6) :
        #chagne frame size
        x, frame = gcv.data.transforms.presets.yolo.transform_test(frame, short=512, max_size=350)
        x = x.as_in_context(self.ctx)
        #get bbox
        class_IDs, scores, bounding_boxs = self.detector(x)
        #get pose
        pose_input, upscale_bbox = detector_to_simple_pose(frame, class_IDs, scores, bounding_boxs,
                                                       output_shape=(128, 96), thr=bbox_thr, ctx=self.ctx)
        if len(upscale_bbox) > 0:
            predicted_heatmap = self.pose_net(pose_input)
            pred_coords, confidence = heatmap_to_coord(predicted_heatmap, upscale_bbox)
        else :
            pred_coords = nd.zeros((1, 17, 2)).as_in_context(self.ctx)
            confidence = nd.zeros((1, 17, 1)).as_in_context(self.ctx)
        
        return pred_coords, confidence, upscale_bbox
    
    def __call__(self, frame, bbox_thr=0.6) :
        return self.predict(frame, bbox_thr)