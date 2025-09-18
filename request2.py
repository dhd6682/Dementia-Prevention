import cv2
import mediapipe as mp
import socket
import math

# 미디어파이프 손 모듈 초기화
mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils

# UDP 클라이언트 설정
udp_ip = "ip-address"  # Unity 또는 다른 수신 장치의 IP 주소
udp_port = "port-num"
sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)

# 웹캡 캡처 설정
cap = cv2.VideoCapture(0)

def calculate_angle(wrist_point, center_point):
    # 중심점과 손목 사이의 벡터와 수직선(12시 방향) 사이의 각도 계산
    vector = (wrist_point[0] - center_point[0], wrist_point[1] - center_point[1])
    reference_vector = (0, -1)  # 기준이 되는 벡터 (12시 방향)

    angle = math.degrees(math.atan2(vector[1], vector[0]) - math.atan2(reference_vector[1], reference_vector[0]))

    # 시계 방향으로 각도 계산
    angle = angle % 360

    return angle

# # OpenCV 창을 이름 지정 및 전체 화면으로 설정
# cv2.namedWindow('Hand Tracking', cv2.WND_PROP_FULLSCREEN)
# cv2.setWindowProperty('Hand Tracking', cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)

with mp_hands.Hands(static_image_mode=False, max_num_hands=2, min_detection_confidence=0.5, min_tracking_confidence=0.5) as hands:
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        # 프레임을 좌우 반전 (플립)
        frame = cv2.flip(frame, 1)

        # 프레임을 RGB로 변환
        image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        image.flags.writeable = False
        results = hands.process(image)
        image.flags.writeable = True
        image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

        if results.multi_hand_landmarks:
            # 화면의 중심점 계산
            height, width, _ = frame.shape
            center_point = (width // 2, height // 2)

            left_wrist_point = None
            right_wrist_point = None

            for hand_landmarks, hand_info in zip(results.multi_hand_landmarks, results.multi_handedness):
                wrist = hand_landmarks.landmark[mp_hands.HandLandmark.WRIST]
                wrist_point = (int(wrist.x * width), int(wrist.y * height))

                if hand_info.classification[0].label == 'Right':
                    right_wrist_point = wrist_point
                    cv2.line(image, center_point, wrist_point, (255, 0, 0), 3)  # 파란색, 분침
                else:
                    left_wrist_point = wrist_point
                    cv2.line(image, center_point, wrist_point, (0, 255, 0), 3)  # 녹색, 시침

                # 미디어파이프 랜드마크 그리기
                mp_drawing.draw_landmarks(
                    image=image,
                    landmark_list=hand_landmarks,
                    connections=mp_hands.HAND_CONNECTIONS
                )

            # 왼손과 오른손의 좌표 및 시간을 한 줄로 결합하여 UDP로 전송
            if left_wrist_point and right_wrist_point:
                hour_angle = calculate_angle(left_wrist_point, center_point)
                minute_angle = calculate_angle(right_wrist_point, center_point)
                time_text = f'{int(hour_angle // 30)}:{int(minute_angle // 6):02d}'

                # 데이터를 하나의 문자열로 구성
                data = f"L,{left_wrist_point[0]},{left_wrist_point[1]},R,{right_wrist_point[0]},{right_wrist_point[1]},T,{time_text}"
                sock.sendto(data.encode('utf-8'), (udp_ip, udp_port))

                # 시간 정보를 화면에 출력
                cv2.putText(image, time_text, (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2, cv2.LINE_AA)

        cv2.imshow('Hand Tracking', image)

        if cv2.waitKey(1) & 0xFF == 27:  # ESC 키를 누르면 종료
            break

cap.release()
sock.close()
cv2.destroyAllWindows()
