from collections import defaultdict, OrderedDict
import logging
import numpy as np
import copy

from detectron2.evaluation import DatasetEvaluator
from detectron2.data import DatasetCatalog, MetadataCatalog
from detectron2.utils import comm


class MyEvaluator(DatasetEvaluator):
    """
    Evaluate cervix detector performace on metrics: mAP, recall, mIOU, FROC (sensitivity 1/8, 1/4, 1/2, 1, 2, 4, 8)
    """
    def __init__(self, dataset_name, cfg, output_folder):
        self.dataset_name = dataset_name
        self._class_names = MetadataCatalog.get(dataset_name).thing_classes
        self._annos = self._parse_annos()
        self._predictions = defaultdict(list)
        self._output_folder = output_folder
        self._result = {}
        self._logger = logging.getLogger('detectron2.evaluation.evaluator')
        self._iou_threshs = list(range(50, 100, 5))
        self._max_dets = list(cfg.TEST.MAX_DETS)

    def _parse_annos(self):
        annos = DatasetCatalog.get(self.dataset_name)
        records = defaultdict(list)
        for annos_a_img in annos:
            image_id = annos_a_img['image_id']
            for anno in annos_a_img['annotations']:
                records[image_id].append({
                    'class': anno['category_id'],
                    'box': anno['bbox']
                })
        return records

    def reset(self):
        self._predictions = defaultdict(list)
        self._result = {}

    def process(self, inputs, outputs):
        for input, output in zip(inputs, outputs):
            image_id = input['image_id']
            instances = output['instances'].to('cpu')
            boxes = instances.pred_boxes.tensor.numpy()
            scores = instances.scores.tolist()
            classes = instances.pred_classes.tolist()
            for box, score, cls in zip(boxes, scores, classes):
                self._predictions[cls].append({
                    'image_id': image_id,
                    'box': box,
                    'score': score
                })

    def evaluate(self):
        all_predictions = comm.gather(self._predictions, dst=0)
        if not comm.is_main_process():
            return
        predictions = defaultdict(list)
        for predictions_per_rank in all_predictions:
            for cls_id, v in predictions_per_rank.items():
                predictions[cls_id].extend(v)
        del all_predictions

        self._logger.info('Evaluation results...')
        predictions = self.sort_predictions(predictions)
        K = len(self._class_names)  # class
        T = len(self._iou_threshs)  # iou thresh
        M = len(self._max_dets)  # max detection per image
        aps = -np.ones((K, T, M))
        ars = -np.ones((K, T, M))
        frocs = -np.ones((K, T, M))
        rec_img_list = -np.ones((K, T, M))
        for k_i, cls_name in enumerate(self._class_names):
            if k_i not in predictions:
                continue
            dts = predictions[k_i]
            gts = self.get_cls_gts(k_i)
            for t_i, thresh in enumerate(self._iou_threshs):  # iou from 0.5 to 0.95, step 0.05
                for m_i, max_det in enumerate(self._max_dets):
                    max_rec, ap, froc, rec_img = self.eval(dts, gts, ovthresh=thresh / 100, max_det=max_det)
                    aps[k_i, t_i, m_i] = ap * 100
                    ars[k_i, t_i, m_i] = max_rec * 100
                    frocs[k_i, t_i, m_i] = froc * 100
                    rec_img_list[k_i, t_i, m_i] = rec_img * 100

        self._result = {
            'aps': aps,
            'ars': ars,
            'frocs': frocs,
            'rec_img_list': rec_img_list
        }

        record = self.summarize()
        result = OrderedDict()
        result['bbox'] = record

        return result

    def summarize(self):
        def _summarize(type, iou_t=None, max_det=100):
            i_str = ' {:<18} @[ IoU={:<9} | maxDets={} ] = {:0.5f}'
            mind = [i for i, mdet in enumerate(self._max_dets) if mdet == max_det]
            if iou_t is None:
                tind = slice(len(self._iou_threshs))
            else:
                tind = [i for i, iou_thresh in enumerate(self._iou_threshs) if iou_thresh == iou_t]
            iou_str = '{:0.2f}:{:0.2f}'.format(0.5, 0.95) if iou_t is None else '{:0.2f}'.format(iou_t)
            det_str = '{:>3d}'.format(max_det) if max_det != float('inf') else 'all'

            if type == 'ap':
                title_str = 'Average Precision'
                metric_res = np.mean(self._result['aps'][:, tind, mind])
            elif type == 'ar':
                title_str = 'Average Recall'
                metric_res = np.mean(self._result['ars'][:, tind, mind])
            elif type == 'froc':
                title_str = 'FROC'
                metric_res = np.mean(self._result['frocs'][:, tind, mind])
            elif type == 'irec':
                title_str = 'Image Recall'
                metric_res = np.mean(self._result['rec_img_list'][:, tind, mind])
            else:
                raise ValueError
            metric_res = float(metric_res)
            self._logger.info(i_str.format(title_str, iou_str, det_str, metric_res))
            return metric_res

        ret = OrderedDict()
        # ap
        for max_det in self._max_dets:
            ret[f'AP_Top{max_det}'] = _summarize(type='ap', iou_t=None, max_det=max_det)
            ret[f'AP50_Top{max_det}'] = _summarize(type='ap', iou_t=50, max_det=max_det)
            ret[f'AP75_Top{max_det}'] = _summarize(type='ap', iou_t=75, max_det=max_det)
        # ar
        for max_det in self._max_dets:
            ret[f'AR_Top{max_det}'] = _summarize(type='ar', iou_t=None, max_det=max_det)
        # froc
        ret[f'FROC'] = _summarize(type='froc', iou_t=None, max_det=self._max_dets[-1])
        ret[f'FROC50'] = _summarize(type='froc', iou_t=50, max_det=self._max_dets[-1])
        ret[f'FROC75'] = _summarize(type='froc', iou_t=75, max_det=self._max_dets[-1])

        # image level recall
        for max_det in self._max_dets:
            ret[f'iRecall_Top{max_det}'] = _summarize(type='irec', iou_t=None, max_det=max_det)
            ret[f'iRecall50_Top{max_det}'] = _summarize(type='irec', iou_t=50, max_det=max_det)
            ret[f'iRecall75_Top{max_det}'] = _summarize(type='irec', iou_t=75, max_det=max_det)

        return ret

    def sort_predictions(self, predictions):
        """
        sort each class's predictions in score descending order, preserve only image_id and detection boxes
        """
        res = {}
        for cls_id, cls_predictions in predictions.items():
            image_ids = [x['image_id'] for x in cls_predictions]
            scores = np.array([x['score'] for x in cls_predictions])
            boxes = np.array([x['box'] for x in cls_predictions]).astype(float)
            # sort by score
            sorted_ind = np.argsort(-scores)
            dt_boxes = boxes[sorted_ind, :]
            dt_image_ids = [image_ids[x] for x in sorted_ind]
            res[cls_id] = {
                'dt_boxes': dt_boxes,
                'dt_image_ids': dt_image_ids
            }

        return res

    def get_cls_gts(self, cls_id):
        # get this class's gt
        gt_recs = {}
        npos = 0
        for image_id, annos in self._annos.items():
            R = [x for x in annos if x['class'] == cls_id]
            boxes = np.array([x['box'] for x in R]).astype(float)
            det = [False] * len(R)
            npos += len(R)
            gt_recs[image_id] = {'boxes': boxes, 'det': det}
        return {'gt_recs': gt_recs, 'npos': npos}

    @ staticmethod
    def eval(dts, gts, ovthresh=0.5, max_det=np.inf):
        """
        eval one class
        """
        gt_recs = copy.deepcopy(gts['gt_recs'])
        dt_boxes = dts['dt_boxes']
        dt_image_ids = dts['dt_image_ids']

        # go down dts and mark TPs and FPs
        # recall = tp / npos
        # preicision = tp / tp + fp
        # ap = roc of P-R
        # froc = sum(recall_i) when fp per img = 1/8, 1/4, 1/2, 1, 2, 4, 8
        # fp_per_img = fp / nimg
        max_det_count = defaultdict(int)
        nimg = len(gt_recs)
        img_m = np.zeros(nimg)  # calculate recall on image level, means patient level recall
        fps_thresh = nimg * np.array([1/8, 1/4, 1/2, 1, 2, 4, 8])
        npos = gts['npos']
        tp = []  # true positive, for recall and precision
        fp = []  # false positive, for precision
        for i in range(len(dt_image_ids)):
            image_id = dt_image_ids[i]
            if max_det_count[image_id] >= max_det:
                continue
            max_det_count[image_id] += 1
            dt_box = dt_boxes[i]
            R = gt_recs[image_id]
            gt_boxes = R['boxes']
            ovmax = -np.inf

            if gt_boxes.size > 0:
                # compute overlaps
                # intersection
                ixmin = np.maximum(gt_boxes[:, 0], dt_box[0])
                iymin = np.maximum(gt_boxes[:, 1], dt_box[1])
                ixmax = np.minimum(gt_boxes[:, 2], dt_box[2])
                iymax = np.minimum(gt_boxes[:, 3], dt_box[3])
                iw = np.maximum(ixmax - ixmin + 1.0, 0.0)
                ih = np.maximum(iymax - iymin + 1.0, 0.0)
                inters = iw * ih

                # union
                uni = (
                        (dt_box[2] - dt_box[0] + 1.0) * (dt_box[3] - dt_box[1] + 1.0)
                        + (gt_boxes[:, 2] - gt_boxes[:, 0] + 1.0) * (gt_boxes[:, 3] - gt_boxes[:, 1] + 1.0)
                        - inters
                )

                overlaps = inters / uni
                ovmax = np.max(overlaps)
                jmax = np.argmax(overlaps)

            if ovmax > ovthresh:  # match
                if not R['det'][jmax]:  # gt hasn't been matched
                    img_m[image_id] = 1
                    tp.append(1.0)
                    fp.append(0.0)
                    R['det'][jmax] = 1
                else:  # gt has been matched
                    tp.append(0.0)
                    fp.append(1.0)
            else:  # not match
                tp.append(0.0)
                fp.append(1.0)

        # compute precision recall
        tp = np.array(tp)
        fp = np.array(fp)
        fp = np.cumsum(fp)
        tp = np.cumsum(tp)
        rec = tp / float(npos)
        # avoid divide by zero in case the first detection matches a difficult
        # ground truth
        prec = tp / np.maximum(tp + fp, np.finfo(np.float64).eps)
        ap = MyEvaluator.voc_ap(rec, prec)

        # find first idx where fp > fp_thresh, append sentinel values at the end
        fp = np.concatenate((fp, [np.inf]))
        fp_idx = [min((fp > x).nonzero()[0][0], len(fp)-2) for x in fps_thresh]
        froc = np.mean([rec[idx] for idx in fp_idx])
        max_rec = rec.max()
        rec_img = np.sum(img_m) / len(img_m)

        return max_rec, ap, froc, rec_img

    @staticmethod
    def voc_ap(rec, prec):
        # correct AP calculation
        # first append sentinel values at the end
        mrec = np.concatenate(([0.0], rec, [1.0]))
        mpre = np.concatenate(([0.0], prec, [0.0]))

        # compute the precision envelope
        for i in range(mpre.size - 1, 0, -1):
            mpre[i - 1] = np.maximum(mpre[i - 1], mpre[i])

        # to calculate area under PR curve, look for points
        # where X axis (recall) changes value
        i = np.where(mrec[1:] != mrec[:-1])[0]

        # and sum (\Delta recall) * prec
        ap = np.sum((mrec[i + 1] - mrec[i]) * mpre[i + 1])

        return ap
