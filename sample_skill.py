import os
from watchfor_skill.skill import Skill
from video_urls import url_bigger_fun
import cv2


class OpencvFrameReader:
    def __init__(self, url, period_sec=1):
        self.url = url
        self.next_timestamp_to_decode = 0
        self.period_sec = period_sec
        self.cap = cv2.VideoCapture(url)

    def get_frame(self):
        while True:
            success, frame = self.cap.read()

            if not success:
                return False, None, None

            else:
                time_sec = self.cap.get(cv2.CAP_PROP_POS_MSEC) / 1000

                if time_sec >= self.next_timestamp_to_decode:
                    self.next_timestamp_to_decode += self.period_sec
                    return True, frame, time_sec


class SampleSkill:
    def __init__(self,
                 frame_reader,
                 detection_threshold=0.6,
                 no_person_alert_duration=5
                 ):
        self.frame_reader = frame_reader
        self.detection_threshold = detection_threshold
        self.no_person_alert_duration = no_person_alert_duration

        skill = Skill(nexus_url=os.environ.get('NEXUS_URL'))
        models = skill.get_public_models()
        yolov4_spec = models['detector_yolov4_512']
        self.yolov4 = yolov4_spec.instantiate()

        self.n_frames_wo_person = 0

    def process_frames(self):
        while True:
            have_frame, frame, time = self.frame_reader.get_frame()

            if have_frame:
                inference = self.yolov4.infer(frame)
                print(f'     time: {time:0.3f}sec, objects: {inference}')

                n_people_detected = len(list(filter(
                    lambda o: o.object_class == 'person' and o.score >= self.detection_threshold, inference.objects)))

                # check if more than 1 person was detected
                if n_people_detected >= 2:
                    print(
                        f'ALERT (time:{time:0.3f}): MORE THAN 1 PERSON DETECTED')

                # check if no person for some consequent frames
                if n_people_detected >= 1:
                    self.n_frames_wo_person = 0
                else:
                    self.n_frames_wo_person += 1
                    if self.n_frames_wo_person >= self.no_person_alert_duration:
                        print(
                            f'ALERT (time:{time:0.3f}): no person detected for {self.n_frames_wo_person} frames')
                        self.n_frames_wo_person = 0

            else:
                return


frame_reader = OpencvFrameReader(url_bigger_fun, period_sec=1)
skill = SampleSkill(frame_reader, detection_threshold=0.5,
                    no_person_alert_duration=5)
skill.process_frames()
