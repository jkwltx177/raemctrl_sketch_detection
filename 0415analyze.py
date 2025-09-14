from ultralytics import RTDETR
import cv2
import numpy as np
import os

class SketchAnalyzer:
    def __init__(self, model_path):
        """
        초기화 함수: RTDETR 모델을 로드하고 클래스 이름과 분석 기능을 설정합니다.
        """
        # RTDETR 모델 로드
        self.model = RTDETR(model_path)
        
        # 클래스 이름 목록
        self.class_names = [
            'person_all', 'head', 'face', 'eye', 'nose', 'mouth', 'ear', 'hair',
            'neck', 'body', 'arm', 'hand', 'hat', 'glasses', 'eyebrow', 'beard',
            'open_mouth_(teeth)', 'muffler', 'tie', 'ribbon', 'ear_muff', 'earring', 'necklace', 'ornament',
            'headdress', 'jewel', 'cigarette'
        ]
        
        # 시각화 컬러 맵
        self.colors = [
            (0, 255, 0),    # 녹색
            (255, 0, 0),    # 파란색
            (0, 0, 255),    # 빨간색
            (255, 255, 0),  # 청록색
            (0, 255, 0),  # 노란색
            (255, 0, 255),  # 마젠타
            (128, 128, 0),  # 올리브
            (0, 128, 128),  # 틸
            (128, 0, 128),  # 퍼플
            (255, 165, 0)   # 오렌지
        ]
    
    def predict(self, image_path, conf_threshold=0.20):
        """
        이미지를 예측하고 결과를 반환합니다.
        """
        # 이미지 로드
        self.image = cv2.imread(image_path)
        self.image_height, self.image_width = self.image.shape[:2]
        self.image_area = self.image_width * self.image_height
        
        # 예측 실행
        self.results = self.model(image_path, conf=conf_threshold)
        
        # 결과 처리
        result = self.results[0]
        self.boxes = result.boxes
        
        # 탐지 결과 저장 구조 생성
        self.detections = {name: [] for name in self.class_names}
        
        # 탐지된 객체 처리
        for box in self.boxes:
            try:
                xyxy = box.xyxy.cpu().numpy()[0]
                conf = box.conf.cpu().numpy()[0]
                cls_id = int(box.cls.cpu().numpy()[0])
                
                if cls_id < len(self.class_names):
                    label = self.class_names[cls_id]
                    x1, y1, x2, y2 = map(int, xyxy)
                    width, height = x2 - x1, y2 - y1
                    self.detections[label].append([x1, y1, x2, y2, width, height, conf])
            except Exception as e:
                print(f"박스 처리 중 오류 발생: {str(e)}")
        
        # 그레이스케일 이미지 준비 (분석용)
        self.gray = cv2.cvtColor(self.image, cv2.COLOR_BGR2GRAY)
        self.edges = cv2.Canny(self.gray, 50, 150)
        
        return self.detections
    
    def analyze(self):
        """
        탐지된 객체들을 분석하고 결과를 반환합니다.
        """
        self.analysis_results = {}
        
        # 전체 크기 분석
        self.analyze_overall_size()
        
        # 머리 크기 기반 기대 비율 설정
        self.setup_expected_proportions()
        
        # 머리 크기 분류
        self.analyze_head_size()
        
        # 얼굴 및 머리-몸통 연결성 분석
        self.analyze_face()
        self.analyze_head_body_connection()
        
        # 신체 부위별 세부 분석
        self.analyze_eyes()
        self.analyze_nose()
        self.analyze_mouth()
        self.analyze_ears()
        self.analyze_hair()
        
        # 그림 전반 형식적 분석
        self.analyze_line_properties()
        self.analyze_symmetry()
        self.analyze_distortion()
        self.analyze_details()
        
        return self.analysis_results
    
    def analyze_overall_size(self):
        """
        전체 인물 크기 및 위치 분석
        """
        if 'person_all' in self.detections and len(self.detections['person_all']) > 0:
            person_box = self.detections['person_all'][0]
            x1, y1, x2, y2 = person_box[:4]
            person_area = person_box[4] * person_box[5]
            
            # 크기 분석
            ratio = person_area / self.image_area
            if ratio < 0.3:
                person_size = "매우 작음"
            elif ratio < 0.6:
                person_size = "작음"
            elif ratio < 0.8:
                person_size = "보통"
            elif ratio < 1.0:
                person_size = "큼"
            else:
                person_size = "매우 큼"
            
            # 수평 치우침 분석
            person_center_x = (x1 + x2) / 2
            image_center_x = self.image_width / 2
            tolerance = self.image_width * 0.05
            
            if person_center_x < image_center_x - tolerance:
                bias_horizontal = "수평.왼쪽으로 치우침"
            elif person_center_x > image_center_x + tolerance:
                bias_horizontal = "수평.오른쪽으로 치우침"
            else:
                bias_horizontal = "수평.중앙"
            
            # 수직 위치 분석
            person_center_y = (y1 + y2) / 2
            image_center_y = self.image_height / 2
            vertical_tolerance = self.image_height * 0.05
            
            if abs(person_center_y - image_center_y) < vertical_tolerance * 0.5:
                bias_vertical = "수직.과도한 정중앙"
            elif person_center_y < image_center_y - vertical_tolerance:
                bias_vertical = "수직.상단"
            elif person_center_y > image_center_y + vertical_tolerance:
                bias_vertical = "수직.하단"
            else:
                bias_vertical = "수직.중앙"
            
            # 가장자리 절단 분석
            cut_off = []
            if y1 <= 0:
                cut_off.append("상 절단")
            if x1 <= 0:
                cut_off.append("좌 절단")
            if x2 >= self.image_width:
                cut_off.append("우 절단")
            cut_off_status = ", ".join(cut_off) if cut_off else "절단 없음"
            
            # 분석 결과 저장
            self.detections['person_all'][0].extend([person_size, bias_horizontal, bias_vertical, cut_off_status])
            self.analysis_results['person_overall'] = {
                'size': person_size,
                'horizontal_position': bias_horizontal,
                'vertical_position': bias_vertical,
                'edge_cutoff': cut_off_status
            }
        else:
            self.analysis_results['person_overall'] = {
                'size': "not 검출됨",
                'horizontal_position': "not 검출됨",
                'vertical_position': "not 검출됨",
                'edge_cutoff': "not 검출됨"
            }
    
    def setup_expected_proportions(self):
        """
        머리 크기 기준으로 다른 신체 부위의 기대 비율 설정
        """
        self.expected = {}
        if 'head' in self.detections and len(self.detections['head']) > 0:
            head_box = self.detections['head'][0]
            head_w, head_h = head_box[4], head_box[5]
            
            self.expected = {
                'eye':   {'w': 0.233 * head_w, 'h': 0.143 * head_h},
                'nose':  {'w': 0.067 * head_w, 'h': 0.143 * head_h},
                'mouth': {'w': 0.333 * head_w, 'h': 0.029 * head_h},
                'ear':   {'w': 0.1   * head_w, 'h': 0.229 * head_h},
                'hair':  {'w': 1.333 * head_w, 'h': 0.229 * head_h},
                'neck':  {'w': 0.067 * head_w, 'h': 0.286 * head_h},
                'face':  {'w': 0.6   * head_w, 'h': 0.7   * head_h},
                'body':  {'w': 1.5   * head_w, 'h': 0.3   * head_h},
                'arm':   {'w': 0.25  * head_w, 'h': 1.2   * head_h},
                'hand':  {'w': 0.15  * head_w, 'h': 0.15  * head_h},
            }
    
    def classify_size(self, detected, expected):
        """
        실제 크기와 기대 크기를 비교하여 분류
        """
        if detected < expected * 0.6:
            return "매우 작음", -2
        elif detected < expected * 0.8:
            return "작음", -1
        elif detected <= expected * 1.2:
            return "평균", 0
        elif detected <= expected * 1.4:
            return "큼", 1
        else:
            return "매우 큼", 2
    
    def combine_status(self, score_w, score_h):
        """
        너비와 높이 점수를 결합하여 종합 크기 상태 반환
        """
        total = score_w + score_h
        if total <= -3:
            return "매우 작음"
        elif total == -2:
            return "작음"
        elif total <= 1:
            return "보통"
        elif total == 2:
            return "큼"
        else:
            return "매우 큼"
    
    def analyze_head_size(self):
        """
        머리 크기 분석
        """
        if 'head' in self.detections and len(self.detections['head']) > 0:
            head_box = self.detections['head'][0]
            head_area = head_box[4] * head_box[5]
            
            if 'person_all' in self.detections and len(self.detections['person_all']) > 0:
                person_box = self.detections['person_all'][0]
                person_area = person_box[4] * person_box[5]
                ratio = head_area / person_area
            else:
                ratio = head_area / self.image_area
            
            if ratio < 0.2:
                head_size_class = "작음"
            elif ratio < 0.35:
                head_size_class = "보통"
            else:
                head_size_class = "큼"
            
            self.analysis_results['head_size'] = head_size_class
        else:
            self.analysis_results['head_size'] = "검출되지 않음"
    
    def analyze_face(self):
        """
        얼굴 방향 분석 (정면, 측면, 뒤통수 등)
        """
        if 'face' in self.detections and len(self.detections['face']) > 0:
            num_eyes = len(self.detections.get('eye', []))
            has_nose = 'nose' in self.detections and len(self.detections['nose']) > 0
            has_mouth = 'mouth' in self.detections and len(self.detections['mouth']) > 0
            
            if num_eyes == 0 and not has_nose and not has_mouth:
                face_status = "뒤통수"
            else:
                face_status = "정면 또는 기타"
            
            self.analysis_results['face_direction'] = face_status
        else:
            self.analysis_results['face_direction'] = "검출되지 않음"
    
    def analyze_head_body_connection(self):
        """
        머리와 몸의 연결성 분석
        """
        if 'head' in self.detections and len(self.detections['head']) > 0 and 'body' in self.detections and len(self.detections['body']) > 0:
            head_box = self.detections['head'][0]
            body_box = self.detections['body'][0]
            gap = body_box[1] - head_box[3]  # y1_body - y2_head
            head_height = head_box[5]
            threshold = 0.1 * head_height
            
            if gap > threshold and 'neck' not in self.detections:
                disconnection_status = "머리와 몸의 단절"
            else:
                disconnection_status = "연결됨"
        elif 'head' in self.detections and len(self.detections['head']) > 0 and ('body' not in self.detections or len(self.detections['body']) == 0):
            disconnection_status = "몸 생략"
        elif 'body' in self.detections and len(self.detections['body']) > 0 and ('head' not in self.detections or len(self.detections['head']) == 0):
            disconnection_status = "머리 생략"
        else:
            disconnection_status = "머리와 몸 모두 미검출"
            
        self.analysis_results['head_body_connection'] = disconnection_status
    
    def analyze_eyes(self):
        """
        눈 분석
        """
        eye_details = {}
        
        if 'eye' in self.detections and len(self.detections['eye']) > 0:
            for i, eye_box in enumerate(self.detections['eye']):
                x1, y1, x2, y2 = eye_box[:4]
                
                # 눈 영역 추출 (범위 확인)
                if y1 < y2 and x1 < x2 and y1 >= 0 and x1 >= 0 and y2 <= self.gray.shape[0] and x2 <= self.gray.shape[1]:
                    eye_roi = self.gray[y1:y2, x1:x2]
                    
                    # 가림 여부 확인
                    obscured = False
                    for label in self.detections:
                        if label != 'eye':
                            for box in self.detections[label]:
                                if x1 < box[2] and x2 > box[0] and y1 < box[3] and y2 > box[1]:
                                    obscured = True
                                    break
                    
                    # 윤곽 분석
                    if eye_roi.size > 0:
                        edges_eye = cv2.Canny(eye_roi, 50, 150)
                        edge_ratio = np.count_nonzero(edges_eye) / (eye_roi.shape[0] * eye_roi.shape[1])
                        contour_status = "윤곽만 묘사" if edge_ratio > 0.1 else "일반 묘사"
                        
                        # 진한 눈동자 분석
                        mean_intensity = np.mean(eye_roi)
                        pupil_status = "진한 눈동자" if mean_intensity < 100 else "일반 눈동자"
                    else:
                        contour_status = "분석 불가"
                        pupil_status = "분석 불가"
                else:
                    obscured = "범위 오류"
                    contour_status = "범위 오류"
                    pupil_status = "범위 오류"
                
                # 부위 크기 분석
                if self.expected and 'eye' in self.expected:
                    status_w, score_w = self.classify_size(eye_box[4], self.expected['eye']['w'])
                    status_h, score_h = self.classify_size(eye_box[5], self.expected['eye']['h'])
                    size_status = self.combine_status(score_w, score_h)
                    eye_box.append(size_status)
                else:
                    size_status = "분석 불가"
                
                eye_details[f'eye_{i}'] = {
                    'obscured': "가림" if obscured else "가림 없음",
                    'contour': contour_status,
                    'pupil': pupil_status,
                    'size': size_status
                }
            
            self.analysis_results['eyes'] = {
                'count': len(self.detections['eye']),
                'details': eye_details
            }
        else:
            self.analysis_results['eyes'] = {
                'count': 0,
                'details': {'eye_0': "검출되지 않음"}
            }
    
    def analyze_nose(self):
        """
        코 분석
        """
        if 'nose' in self.detections and len(self.detections['nose']) > 0:
            nose_box = self.detections['nose'][0]
            w, h = nose_box[4], nose_box[5]
            
            # 부위 크기 분석
            if self.expected and 'nose' in self.expected:
                status_w, score_w = self.classify_size(w, self.expected['nose']['w'])
                status_h, score_h = self.classify_size(h, self.expected['nose']['h'])
                size_status = self.combine_status(score_w, score_h)
                nose_box.append(size_status)
            else:
                size_status = "분석 불가"
                
            # 코 형태 분석
            if h > 0:
                aspect_ratio = w / h
                if aspect_ratio < 0.5:
                    nose_shape = "길쭉한 코"
                elif aspect_ratio > 1.0:
                    nose_shape = "넓은 코"
                else:
                    nose_shape = "일반적인 코"
            else:
                nose_shape = "측정 불가능"
                
            self.analysis_results['nose'] = {
                'shape': nose_shape,
                'size': size_status
            }
        else:
            self.analysis_results['nose'] = {
                'shape': "검출되지 않음",
                'size': "검출되지 않음"
            }
    
    def analyze_mouth(self):
        """
        입 분석
        """
        if 'mouth' in self.detections and len(self.detections['mouth']) > 0:
            mouth_box = self.detections['mouth'][0]
            x1, y1, x2, y2 = mouth_box[:4]
            w, h = mouth_box[4], mouth_box[5]
            
            # 부위 크기 분석
            if self.expected and 'mouth' in self.expected:
                status_w, score_w = self.classify_size(w, self.expected['mouth']['w'])
                status_h, score_h = self.classify_size(h, self.expected['mouth']['h'])
                size_status = self.combine_status(score_w, score_h)
                mouth_box.append(size_status)
            else:
                size_status = "분석 불가"
            
            # 입 형태 분석
            if y1 < y2 and x1 < x2 and y1 >= 0 and x1 >= 0 and y2 <= self.gray.shape[0] and x2 <= self.gray.shape[1]:
                mouth_roi = self.gray[y1:y2, x1:x2]
                if mouth_roi.size > 0 and mouth_roi.shape[0] > 0 and mouth_roi.shape[1] > 0:
                    edges_mouth = cv2.Canny(mouth_roi, 50, 150)
                    
                    if self.expected and 'mouth' in self.expected:
                        h_ratio = h / self.expected['mouth']['h']
                        if h_ratio > 1.5:
                            mouth_shape = "벌림"
                        else:
                            # 곡률 분석
                            if mouth_roi.shape[0] > 1:  # 행이 2개 이상인지 확인
                                top_edge = np.sum(edges_mouth[:mouth_roi.shape[0]//2, :])
                                bottom_edge = np.sum(edges_mouth[mouth_roi.shape[0]//2:, :])
                                if abs(top_edge - bottom_edge) < 100:
                                    mouth_shape = "일직선"
                                elif top_edge > bottom_edge:
                                    mouth_shape = "웃음"
                                else:
                                    mouth_shape = "비웃음"
                            else:
                                mouth_shape = "분석 불가 (너무 얇음)"
                    else:
                        mouth_shape = "분석 불가 (기준 없음)"
                else:
                    mouth_shape = "유효하지 않은 이미지"
            else:
                mouth_shape = "유효하지 않은 경계 상자"
            
            self.analysis_results['mouth'] = {
                'shape': mouth_shape,
                'size': size_status
            }
        else:
            self.analysis_results['mouth'] = {
                'shape': "검출되지 않음",
                'size': "검출되지 않음"
            }
    
    def analyze_ears(self):
        """
        귀 분석
        """
        ear_details = {}
        
        if 'ear' in self.detections and len(self.detections['ear']) > 0:
            for i, ear_box in enumerate(self.detections['ear']):
                x1, y1, x2, y2 = ear_box[:4]
                w, h = ear_box[4], ear_box[5]
                
                # 부위 크기 분석
                if self.expected and 'ear' in self.expected:
                    status_w, score_w = self.classify_size(w, self.expected['ear']['w'])
                    status_h, score_h = self.classify_size(h, self.expected['ear']['h'])
                    size_status = self.combine_status(score_w, score_h)
                    ear_box.append(size_status)
                else:
                    size_status = "분석 불가"
                
                # 귀걸이 탐지
                earring_detected = False
                for label in self.detections:
                    if label == 'earring':
                        for box in self.detections[label]:
                            bx1, by1, bx2, by2 = box[:4]
                            if (abs((bx1 + bx2) / 2 - (x1 + x2) / 2) < w and
                                abs((by1 + by2) / 2 - (y1 + y2) / 2) < h):
                                earring_detected = True
                                break
                
                ear_details[f'ear_{i}'] = {
                    'size': size_status,
                    'earring': "귀걸이 착용" if earring_detected else "귀걸이 없음"
                }
            
            self.analysis_results['ears'] = {
                'count': len(self.detections['ear']),
                'details': ear_details
            }
        else:
            self.analysis_results['ears'] = {
                'count': 0,
                'details': {'ear_0': "검출되지 않음"}
            }
    
    def analyze_hair(self):
        """
        머리카락 분석
        """
        if 'hair' in self.detections and len(self.detections['hair']) > 0:
            hair_box = self.detections['hair'][0]
            x1, y1, x2, y2 = hair_box[:4]
            
            # 유효한 좌표인지 확인
            if y1 < y2 and x1 < x2 and y1 >= 0 and x1 >= 0 and y2 <= self.gray.shape[0] and x2 <= self.gray.shape[1]:
                hair_roi = self.gray[y1:y2, x1:x2]
                
                if hair_roi.size > 0:
                    # 숱 분석
                    hair_density = np.mean(hair_roi)
                    volume_status = "많은 숱" if hair_density < 150 else "적은 숱"
                    
                    # 세부 묘사 분석
                    edges_hair = cv2.Canny(hair_roi, 50, 150)
                    edge_density_hair = np.count_nonzero(edges_hair) / (hair_roi.shape[0] * hair_roi.shape[1])
                    detail_status = "세부 묘사 있음" if edge_density_hair > 0.05 else "세부 묘사 없음"
                else:
                    volume_status = "분석 불가"
                    detail_status = "분석 불가"
            else:
                volume_status = "좌표 오류"
                detail_status = "좌표 오류"
            
            # 부위 크기 분석
            if self.expected and 'hair' in self.expected:
                status_w, score_w = self.classify_size(hair_box[4], self.expected['hair']['w'])
                status_h, score_h = self.classify_size(hair_box[5], self.expected['hair']['h'])
                size_status = self.combine_status(score_w, score_h)
                hair_box.append(size_status)
            else:
                size_status = "분석 불가"
            
            self.analysis_results['hair'] = {
                'volume': volume_status,
                'detail': detail_status,
                'size': size_status
            }
        else:
            self.analysis_results['hair'] = {
                'volume': "검출되지 않음",
                'detail': "검출되지 않음",
                'size': "검출되지 않음"
            }
    
    def analyze_line_properties(self):
        """
        선 특성 분석 (필압, 선의 길이, 모양 등)
        """
        # 필압 분석
        edge_density = np.count_nonzero(self.edges) / (self.image_height * self.image_width)
        pressure_status = "강한 필압" if edge_density > 0.05 else "약한 필압"
        
        # 필압 변화 분석
        regions = [self.edges[:self.image_height//2, :], self.edges[self.image_height//2:, :]]
        densities = [np.count_nonzero(region) / (region.shape[0] * region.shape[1]) for region in regions]
        pressure_var = np.std(densities)
        pressure_change = "과도한 변화" if pressure_var > 0.02 else "적당한 변화"
        
        # 선 분석
        lines = cv2.HoughLinesP(self.edges, 1, np.pi/180, threshold=50, minLineLength=30, maxLineGap=10)
        
        if lines is not None:
            lengths = [np.linalg.norm((line[0][0]-line[0][2], line[0][1]-line[0][3])) for line in lines]
            avg_line_length = np.mean(lengths)
            line_status = "전체적 강한 선" if avg_line_length > 100 else "일부분 강한 선"
            line_length_status = "길다" if avg_line_length > 100 else "짧다"
            
            # 지면선 존재 여부
            ground_line_exists = False
            for line in lines:
                x1_l, y1_l, x2_l, y2_l = line[0]
                dx = x2_l - x1_l
                dy = y2_l - y1_l
                if dx != 0:  # 0으로 나누기 방지
                    angle = np.degrees(np.arctan2(dy, dx))
                    if abs(angle) < 10 or abs(angle - 180) < 10:
                        if y1_l > self.image_height * 0.8 and y2_l > self.image_height * 0.8:
                            ground_line_exists = True
                            break
        else:
            line_status = "약한 선"
            line_length_status = "정보 없음"
            ground_line_exists = False
        
        # 곡선/직선 분석
        contours, _ = cv2.findContours(self.edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        curvature_scores = []
        
        for cnt in contours:
            if len(cnt) > 4:  # 충분한 포인트가 있는지 확인
                epsilon = 0.01 * cv2.arcLength(cnt, True)
                approx = cv2.approxPolyDP(cnt, epsilon, True)
                curvature_scores.append(0 if len(approx) <= 3 else 1)
        
        if curvature_scores:
            shape_status = "직선적" if np.mean(curvature_scores) < 0.5 else "곡선적"
        else:
            shape_status = "정보 없음"
        
        self.analysis_results['line_properties'] = {
            'pressure': pressure_status,
            'pressure_change': pressure_change,
            'line_quality': line_status,
            'line_length': line_length_status,
            'shape_type': shape_status,
            'ground_line': "있음" if ground_line_exists else "없음"
        }
    
    def analyze_symmetry(self):
        """
        대칭성 분석
        """
        if 'head' in self.detections and len(self.detections['head']) > 0:
            head_box = self.detections['head'][0]
            x1_h, y1_h, x2_h, y2_h = head_box[:4]
            
            # 헤드 이미지가 유효한지 확인
            if y2_h > y1_h and x2_h > x1_h and y1_h >= 0 and x1_h >= 0 and y2_h <= self.image.shape[0] and x2_h <= self.image.shape[1]:
                head_img = self.image[y1_h:y2_h, x1_h:x2_h]
                
                # 이미지가 비어있지 않은지 확인
                if head_img.size > 0 and head_img.shape[0] > 0 and head_img.shape[1] > 0:
                    try:
                        head_img_flipped = cv2.flip(head_img, 1)
                        diff = cv2.absdiff(head_img, head_img_flipped)
                        
                        # diff가 유효한지 확인
                        if diff is not None and diff.size > 0:
                            mean_diff = np.mean(diff)
                            symmetry_status = "대칭성이 높음" if mean_diff < 20 else "대칭성이 낮음"
                        else:
                            symmetry_status = "대칭 계산 실패"
                    except Exception as e:
                        symmetry_status = f"대칭 계산 오류: {str(e)}"
                else:
                    symmetry_status = "이미지 추출 실패"
            else:
                symmetry_status = "유효하지 않은 경계 상자"
        else:
            symmetry_status = "머리 검출 안됨"
        
        self.analysis_results['symmetry'] = symmetry_status
    
    def analyze_distortion(self):
        """
        왜곡 분석
        """
        # 왜곡 분석 방법 1: 특정 부위별 왜곡
        distortion_status = {}
        for part in self.expected.keys():
            if part in self.detections and len(self.detections[part]) > 0:
                for box in self.detections[part]:
                    try:
                        w, h = box[4], box[5]
                        expected_w = self.expected[part]['w']
                        expected_h = self.expected[part]['h']
                        
                        if expected_w <= 0 or expected_h <= 0:
                            distortion = "기준값 오류"
                        else:
                            w_ratio = w / expected_w
                            h_ratio = h / expected_h
                            
                            if w_ratio > 2.0 or h_ratio > 2.0:
                                distortion = "극단적인 왜곡"
                            elif w_ratio > 1.5 or h_ratio > 1.5:
                                distortion = "일반적인 왜곡"
                            else:
                                distortion = "왜곡 없음"
                        
                        distortion_status[part] = distortion
                    except Exception as e:
                        distortion_status[part] = f"계산 오류: {str(e)}"
            else:
                distortion_status[part] = "검출되지 않음"
        
        # 왜곡 분석 방법 2: 전체적 왜곡
        extreme = 0
        total = 0
        for part in self.expected.keys():
            if part in self.detections:
                for box in self.detections[part]:
                    if len(box) > 7 and box[7] in ["매우 작음", "매우 큼"]:
                        extreme += 1
                    total += 1
        
        if total == 0:
            overall_distortion = "정보 없음"
        else:
            ratio = extreme / total
            if ratio > 0.5:
                overall_distortion = "극단적인 왜곡"
            elif ratio > 0.2:
                overall_distortion = "일반적인 왜곡"
            else:
                overall_distortion = "정상"
        
        self.analysis_results['distortion'] = {
            'overall': overall_distortion,
            'details': distortion_status
        }
    
    def analyze_details(self):
        """
        세부 묘사 분석
        """
        # 오브 특징점 검출기를 사용한 세부 묘사 분석
        orb = cv2.ORB_create()
        keypoints = orb.detect(self.gray, None)
        num_keypoints = len(keypoints)
        density = num_keypoints / self.image_area
        
        if density > 0.0005:
            detail_status = "과도한 세부 묘사"
        elif density < 0.0001:
            detail_status = "부족한 세부 묘사"
        else:
            detail_status = "보통"
        
        # 불필요한 내용 분석
        mask = np.zeros_like(self.gray)
        for label in self.detections:
            for box in self.detections[label]:
                x1, y1, x2, y2 = box[:4]
                cv2.rectangle(mask, (x1, y1), (x2, y2), 255, -1)
        
        keypoints_in_parts = len(orb.detect(cv2.bitwise_and(self.gray, self.gray, mask=mask)))
        unnecessary_ratio = (num_keypoints - keypoints_in_parts) / num_keypoints if num_keypoints > 0 else 0
        
        if unnecessary_ratio > 0.3:
            unnecessary_content = "불필요한 내용 과도함"
        else:
            unnecessary_content = "불필요한 내용 없음"
        
        # 움직임 표현 분석
        laplacian_var = cv2.Laplacian(self.gray, cv2.CV_64F).var()
        if laplacian_var < 100:
            movement_status = "과도한 움직임 표현"
        elif laplacian_var > 300:
            movement_status = "움직임 부족"
        else:
            movement_status = "적당한 움직임"
        
        self.analysis_results['detail_analysis'] = {
            'detail_level': detail_status,
            'unnecessary_content': unnecessary_content,
            'movement': movement_status
        }
    
    def visualize(self, show_image=True, save_path=None):
        """
        분석 결과를 시각화하여 이미지에 표시하고 저장합니다.
        """
        # 사본 생성 (원본 이미지 보존)
        output_img = self.image.copy()
        
        # 각 검출된 객체에 대해 바운딩 박스와 정보 표시
        for label in self.detections:
            for i, box in enumerate(self.detections[label]):
                x1, y1, x2, y2, width, height, conf = box[:7]
                # 좌표를 정수로 변환
                x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
                
                # 클래스 색상 선택
                cls_id = self.class_names.index(label) if label in self.class_names else 0
                color_idx = cls_id % len(self.colors)
                color = self.colors[color_idx]
                
                # 박스 그리기
                cv2.rectangle(output_img, (x1, y1), (x2, y2), color, 2)
                
                # 표시할 텍스트 준비
                if label == 'person_all' and len(box) > 10:
                    status = box[7]  # person_size
                elif len(box) > 7:
                    status = box[7]  # 부위별 크기 상태
                else:
                    status = ""
                
                text1 = f"{label} {conf:.2f}"
                text2 = f"{status}" if status else ""
                
                # 텍스트 크기 계산
                font = cv2.FONT_HERSHEY_SIMPLEX
                scale = 0.5
                thickness = 1
                (text1_width, text1_height), _ = cv2.getTextSize(text1, font, scale, thickness)
                if text2:
                    (text2_width, text2_height), _ = cv2.getTextSize(text2, font, scale, thickness)
                    box_height = text1_height + text2_height + 5
                    box_width = max(text1_width, text2_width)
                else:
                    box_height = text1_height
                    box_width = text1_width
                
                # 텍스트 배경 그리기
                cv2.rectangle(output_img, (x1, y1 - box_height - 5), (x1 + box_width, y1), color, -1)
                
                # 텍스트 그리기
                cv2.putText(output_img, text1, (x1, y1 - (box_height - text1_height) - 5), font, scale, (255, 255, 255), thickness)
                if text2:
                    cv2.putText(output_img, text2, (x1, y1 - 5), font, scale, (255, 255, 255), thickness)
        
        # 이미지 표시
        if show_image:
            cv2.imshow("Sketch Analysis", output_img)
            cv2.waitKey(0)
            cv2.destroyAllWindows()
        
        # 이미지 저장
        if save_path:
            cv2.imwrite(save_path, output_img)
        else:
            # 저장 경로가 지정되지 않은 경우 자동 생성
            output_file = "analyzed_" + os.path.basename(self.image_path if hasattr(self, 'image_path') else "output.jpg")
            cv2.imwrite(output_file, output_img)
        
        return output_img
    
    def print_analysis_results(self):
        """
        분석 결과를 콘솔에 출력합니다.
        """
        print("\n=== 스케치 분석 결과 ===")
        
        # 1. 전체 인물 분석
        print("\n## 전체 인물 분석")
        if 'person_overall' in self.analysis_results:
            person = self.analysis_results['person_overall']
            print(f"크기: {person['size']}")
            print(f"위치: {person['horizontal_position']}, {person['vertical_position']}")
            print(f"절단: {person['edge_cutoff']}")
        
        # 2. 머리 분석
        print("\n## 머리 분석")
        if 'head_size' in self.analysis_results:
            print(f"머리 크기: {self.analysis_results['head_size']}")
        if 'head_body_connection' in self.analysis_results:
            print(f"머리-몸 연결: {self.analysis_results['head_body_connection']}")
        if 'face_direction' in self.analysis_results:
            print(f"얼굴 방향: {self.analysis_results['face_direction']}")
        
        # 3. 눈, 코, 입, 귀 분석
        print("\n## 얼굴 세부 분석")
        if 'eyes' in self.analysis_results:
            eyes = self.analysis_results['eyes']
            print(f"눈 개수: {eyes['count']}")
            for key, value in eyes['details'].items():
                if isinstance(value, dict):
                    print(f"  {key}: {', '.join([f'{k}={v}' for k, v in value.items()])}")
                else:
                    print(f"  {key}: {value}")
        
        if 'nose' in self.analysis_results:
            nose = self.analysis_results['nose']
            print(f"코: 형태={nose['shape']}, 크기={nose['size']}")
        
        if 'mouth' in self.analysis_results:
            mouth = self.analysis_results['mouth']
            print(f"입: 형태={mouth['shape']}, 크기={mouth['size']}")
        
        if 'ears' in self.analysis_results:
            ears = self.analysis_results['ears']
            print(f"귀 개수: {ears['count']}")
            for key, value in ears['details'].items():
                if isinstance(value, dict):
                    print(f"  {key}: {', '.join([f'{k}={v}' for k, v in value.items()])}")
                else:
                    print(f"  {key}: {value}")
        
        if 'hair' in self.analysis_results:
            hair = self.analysis_results['hair']
            print(f"머리카락: 숱={hair['volume']}, 세부묘사={hair['detail']}, 크기={hair['size']}")
        
        # 4. 형식적 특성 분석
        print("\n## 형식적 특성 분석")
        if 'line_properties' in self.analysis_results:
            line = self.analysis_results['line_properties']
            print(f"선 특성: 필압={line['pressure']}, 변화={line['pressure_change']}")
            print(f"        선질={line['line_quality']}, 길이={line['line_length']}")
            print(f"        형태={line['shape_type']}, 지면선={line['ground_line']}")
        
        if 'symmetry' in self.analysis_results:
            print(f"대칭성: {self.analysis_results['symmetry']}")
        
        if 'distortion' in self.analysis_results:
            distortion = self.analysis_results['distortion']
            print(f"왜곡: 전체={distortion['overall']}")
            print("  세부 왜곡:")
            for part, status in distortion['details'].items():
                print(f"    {part}: {status}")
        
        if 'detail_analysis' in self.analysis_results:
            detail = self.analysis_results['detail_analysis']
            print(f"세부 묘사: {detail['detail_level']}")
            print(f"불필요한 내용: {detail['unnecessary_content']}")
            print(f"움직임 표현: {detail['movement']}")
        
        print("\n=== 분석 완료 ===")


def main():
    # 모델 경로 설정
    model_path = "C:/capstone/yolo/runs/detect/train25/weights/best.pt"  # 경로를 실제 모델 위치로 변경하세요
    
    # 이미지 경로 설정
    image_path = "C:/capstone/yolo/0428_001.jpg"  # 경로를 실제 이미지 위치로 변경하세요
    
    # 스케치 분석기 초기화
    analyzer = SketchAnalyzer(model_path)
    
    # 이미지 예측 및 분석
    analyzer.predict(image_path)
    analyzer.analyze()
    
    # 분석 결과 출력
    analyzer.print_analysis_results()
    
    # 결과 시각화 및 저장
    analyzer.visualize(show_image=True)
    
    print(f"분석이 완료되었습니다. 결과 이미지가 저장되었습니다.")


if __name__ == "__main__":
    main()