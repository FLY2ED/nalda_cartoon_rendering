# Nalda Cartoon Rendering
Nalda Cartoon Rendering: 강력한 이미지-만화 변환기

## 소개
Nalda Cartoon Rendering은 일반 이미지를 만화 스타일로 변환하는 프로그램입니다. OpenCV의 이미지 처리 기술을 활용하여 사진에 만화적인 효과를 적용합니다.

## 주요 기능
- 이미지를 만화 스타일로 변환
- 다양한 파라미터 조정 가능 (엣지 강도, 색상 평활화 등)
- 사용자 친화적인 GUI 인터페이스
- 결과 이미지 저장 기능

## 사용 방법
1. 프로그램을 실행합니다.
2. '이미지 열기' 버튼을 클릭하여 변환할 이미지를 선택합니다.
3. 슬라이더를 조정하여 만화 효과의 강도를 조절합니다.
4. '변환' 버튼을 클릭하여 이미지를 만화 스타일로 변환합니다.
5. '저장' 버튼을 클릭하여 결과 이미지를 저장합니다.

## 결과 예시

### 잘 변환된 예시
![잘 변환된 예시](examples/good_example.jpg)

### 변환이 어려운 예시
![변환이 어려운 예시](examples/challenging_example.jpg)

## 알고리즘 한계점

현재 구현된 알고리즘은 다음과 같은 한계점이 있습니다:

1. **복잡한 배경**: 배경이 복잡하거나 세부 사항이 많은 이미지에서는 만화 효과가 잘 표현되지 않습니다.
2. **저조도 이미지**: 어두운 이미지나 대비가 낮은 이미지에서는 엣지 검출이 제대로 작동하지 않아 만화 효과가 약해집니다.
3. **얼굴 인식 부재**: 현재 알고리즘은 얼굴 특징을 특별히 처리하지 않아 인물 사진에서 표정이나 특징이 과도하게 단순화될 수 있습니다.
4. **색상 제한**: 현재 구현은 색상 양자화가 제한적이어서 다양한 만화 스타일을 표현하는 데 한계가 있습니다.

## 향후 개선 방향
- 딥러닝 기반 스타일 변환 기법 도입
- 얼굴 인식 기능 추가로 인물 사진 처리 개선
- 다양한 만화 스타일 프리셋 제공
- 실시간 비디오 처리 지원

## 요구 사항
- Python 3.6 이상
- OpenCV
- NumPy
- tkinter (GUI용)
- PIL/Pillow (이미지 처리용)
