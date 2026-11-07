# Báo cáo bài thực hành Ngày 1 – Đọc prediction YOLO11 để hiểu định dạng nhãn

**Ngày chạy:** 7/11/2026

**Checkpoint:** `yolo11n-cls.pt`, `yolo11n.pt`, `yolo11n-seg.pt`

**Ngưỡng detection/segmentation dùng khi tạo JSON:** `0.35` 

**Thay đổi so với notebook nguồn:** Không 


## 1. Phân loại ảnh

Nguồn: `classification_predictions.json`, `classification_top5.png`, ảnh `sample_id = traffic`, `rank = 1`.

1. Kết quả dự đoán Record 1 có các thông tin:
class_id = 468
class_name = cab
rank = 1
score = 0.510915
taxonomy_name = ImageNet-1K
2. Record này nói về **cả bức ảnh**. Trong JSON chỉ có một nhãn cho mỗi ảnh (kèm `image_width`, `image_height`), không có tọa độ box hay polygon nào chỉ ra vị trí một vật thể. Bức ảnh đường phố có rất nhiều xe buýt và ô tô, nhưng model chỉ trả về một lớp cho toàn cảnh.
3. Danh sách tên lớp **đi kèm taxonomy/checkpoint** (ImageNet-1K, 1000 lớp), model không tự nghĩ ra. Model chỉ chọn trong danh sách có sẵn. Vì vậy ảnh này bị gán “cab” dù phần lớn xe trong ảnh là xe buýt: ImageNet-1K không có lớp “cảnh giao thông” hay “đường phố đông xe”.
4. Cần ghi cả ba vì tên lớp một mình dễ mơ hồ hoặc bị đổi. `class_id` chỉ có nghĩa trong một taxonomy cụ thể (id 468 ở ImageNet-1K khác id 468 ở bộ khác). `class_name` giúp người đọc hiểu nhanh, và `taxonomy_name` cho biết đang dùng danh sách nào. Có đủ ba thứ thì nhãn mới truy vết và so sánh được giữa các người gán nhãn, các lần huấn luyện.
5. Guideline cần nói rõ **tiêu chí chọn lớp chính** khi ảnh có nhiều vật/chủ đề: chọn vật chiếm diện tích lớn nhất, vật ở trung tâm, hay chủ đề chính của ảnh. Ngoài ra cần nói rõ phải làm gì khi không lớp nào phù hợp (vd. chọn lớp gần nhất, hay đánh dấu “không rõ” và escalate).
6. Vì `score` chỉ là mức “tin” của model, không do người kiểm chứng. Ảnh `traffic` có score 0.51 cho “cab”, nhưng model vẫn có thể sai: top 5 đều là loại xe khác nhau (minibus 0.16, police_van 0.09...) và không lớp nào mô tả đúng cảnh nhiều xe buýt. Ground truth phải do người gán theo guideline rồi được QC. Điểm của model chỉ dùng để tham khảo hoặc để ưu tiên ca cần kiểm tra.

## 2. Phát hiện vật thể

Nguồn: `detection_predictions.json`, `detection_predictions.png`, ảnh `sample_id = kitchen`.

1. Kết quả dự đoán Record có các thông tin:
class_name = person
score = 0.912625
bbox_xyxy = [385.33, 69.24, 498.92, 348.92]
bbox_width = 113.58
bbox_height = 279.68
2. Hộp có góc **trên-trái** tại (x=385.33, y=69.24) và góc **dưới-phải** tại (x=498.92, y=348.92), tính bằng pixel từ góc trên-trái của ảnh (ảnh kitchen rộng 640, cao 427). Hộp rộng 113.58 pixel (498.92 − 385.33) và cao 279.68 pixel (348.92 − 69.24). Trên ảnh, đó là hộp bao quanh người đàn ông đứng quay lưng.
3. Hai ngưỡng, đếm từ `detection_predictions.json` (ảnh `kitchen`):
   - Ngưỡng **0.35** (ngưỡng đã dùng khi tạo JSON): **11** prediction.
   - Ngưỡng **0.60**: **6** prediction (person 0.91, bowl 0.72, bowl 0.70, oven 0.69, oven 0.63, person 0.61).
   - Giữ lại các hộp có score ≥ ngưỡng, nên ngưỡng thấp hơn giữ nhiều hộp hơn. (Số ở ngưỡng 0.20 cần lấy từ output ô so sánh của notebook, vì JSON đã lọc từ 0.35.)
4. Khi hạ ngưỡng thì có thêm nhiều hộp score thấp, nên **reviewer** phải kiểm tra nhiều hộp hơn: loại hộp thừa/trùng, sửa nhãn lớp sai, và đồng thời tìm vật còn sót. Trên ảnh `kitchen`, các hộp `bowl` chồng lên nhau ở góc dưới trái (bowl 0.50, 0.38, 0.72) là ví dụ về việc phải quyết định hộp nào đúng, hộp nào trùng.
5. **Quy tắc box:** vẽ một hộp chữ nhật khít quanh phần *nhìn thấy* của mỗi vật thể (sát mép, không chừa khoảng trống thừa, không cắt mất phần vật), mỗi vật một hộp, và không vẽ hộp trùng cho cùng một vật.
6. Vật bị cắt mép: hộp `person` score 0.61 ở mép trái ảnh (bbox bắt đầu từ x≈0.12), chỉ thấy phần tay/ống tay áo, phần còn lại ngoài khung ảnh. Mình sẽ **đưa lên hỏi Lab Coach (escalation)**: quy tắc “vẽ hộp quanh phần nhìn thấy” có thể đưa vào guideline, nhưng việc chỉ thấy một bàn tay thì có tính là “person” hay không là quyết định về phạm vi nhãn (cần ngưỡng tối thiểu phần nhìn thấy), người gán không được tự đoán. Sau khi Coach trả lời thì mới viết thành quy tắc.

## 3. Phân đoạn theo từng đối tượng

Nguồn: `segmentation_predictions.json`, `segmentation_prediction.png`, ảnh `sample_id = kitchen`.

1. Kết quả dự đoán Record có các thông tin:
instance_id = kitchen-001
class_name = person
score = 0.899318
polygon_point_count = 348
5 điểm đầu tiên của polygon_xy: [446, 70], [445, 71], [444, 71], [443, 72], [442, 72].
2. Box chỉ cho biết vật nằm trong một hình chữ nhật. Với record này, hộp rộng khoảng 113 × 282 pixel gồm cả nền phía sau người. Polygon thì đi theo **đường viền thật** của vật (đầu, vai, tay chống hông, chân...), nên cho biết hình dạng và phần diện tích mà vật thật sự chiếm, ít lẫn nền hơn. Số điểm 348 cũng cho thấy hình dạng người phức tạp hơn nhiều so với 4 góc của box.
3. `instance_id` (vd. `kitchen-001`) là mã để phân biệt **từng đối tượng riêng** trong một ảnh, kể cả khi cùng lớp (trong ảnh kitchen có nhiều `bowl`, mỗi cái có id khác nhau: kitchen-002, 003, 006, 010). Nó **không phải class ID** vì class ID (vd. person = 0) là mã của loại vật trong taxonomy, giống nhau cho mọi người. Nó cũng **không phải tracking ID** vì tracking ID dùng để theo dõi cùng một vật qua nhiều khung hình video, còn `instance_id` chỉ có nghĩa trong một ảnh.
4. **Quy tắc biên mask:** vẽ polygon đi sát đường viền ngoài của phần nhìn thấy của vật; không để lọt nền vào trong và không cắt mất phần vật; mỗi pixel chỉ thuộc về một đối tượng, và khi hai vật chồng nhau thì phần chồng thuộc về vật nằm phía trước.
5. Chỗ khó trên ảnh `kitchen`: ở góc dưới trái, mask `dining table` (score 0.58) và mask `oven` (0.49) chồng lên nhau, cùng với `person` (0.43) và các `bowl`; mask bàn còn lan sang cả phần kệ phía dưới. Cách xử lý: dùng **guideline**, vì đây là tình huống lặp lại thường xuyên và có thể quy định chung theo quy tắc ở mục 4 (vật phía trước giữ pixel của phần chồng). Nếu có phần bị che mà không xác định được thuộc về vật nào, hoặc không rõ đó là bàn hay bếp, thì mới escalate cho Lab Coach.

## 4. Vòng đời và kiểm tra chất lượng

`ảnh thô → guideline → ground truth → huấn luyện → prediction → QC/rework`

| Tác vụ | Ground truth trông như thế nào? (đơn vị nhãn) | Lỗi hoặc chỗ khó quyết định bạn thấy trên ảnh/JSON | Annotator làm gì? | Reviewer xem gì? |
| --- | --- | --- | --- | --- |
| Phân loại ảnh | Một nhãn đại diện cho toàn bộ ảnh, gồm class_id, class_name và taxonomy_name | Ảnh `traffic` có nhiều xe buýt và ô tô nhưng model dự đoán `cab` ở vị trí top-1 với score 0.51. ImageNet-1K không có lớp mô tả toàn bộ cảnh giao thông. | Chọn nhãn đại diện cho ảnh theo guideline. Nếu không có lớp phù hợp thì báo Lab Coach, không tự ý chọn nhãn | Kiểm tra nhãn có phù hợp với nội dung ảnh và đúng taxonomy không. Không dựa hoàn toàn vào score của model |
| Phát hiện vật thể | Mỗi đối tượng có một bounding box `bbox_xyxy` và một nhãn lớp `class_name` | Ảnh `kitchen` có các box `bowl` chồng lên nhau. Một đối tượng `person` ở mép trái chỉ nhìn thấy một phần cánh tay | Vẽ box sát phần nhìn thấy của vật thể, chọn đúng nhãn và tránh tạo box trùng. Trường hợp không rõ thì hỏi Lab Coach | Kiểm tra box có bao quanh đúng vật thể không, có bị thiếu, thừa hoặc trùng không; kiểm tra cả nhãn lớp |
| Instance segmentation | Mỗi đối tượng có một polygon/mask, nhãn lớp và `instance_id` riêng | Mask `dining` table bị lan sang phần kệ phía dưới và chồng lấn với `oven`, `person` và các `bowl` | Vẽ polygon theo đường viền vật thể, tách riêng từng đối tượng và xử lý vùng che khuất theo guideline | Kiểm tra đường biên mask, phần nền lọt vào, vùng chồng lấn và việc phân biệt ID giữa các đối tượng |

## 5. An toàn dữ liệu

1. Chỉ dùng ba ảnh COCO 2017 (CC BY 2.0) mà bài lab cung cấp, không đưa dữ liệu cá nhân (họ tên, MSSV, email, số điện thoại...) hay ảnh ngoài phạm vi vào báo cáo hoặc output.
2. Nếu gặp ảnh hoặc dữ liệu không đúng phạm vi, dừng lại, không xử lý hay tải lên, và báo cho Lab Coach (giảng viên).

## 6. Checklist trước khi nộp link repo

Kiểm tra trên GitHub trong thư mục `report/`:

- [ ] `REPORT.md` (file này đã điền)
- [ ] `day1_lab_outputs/classification_predictions.json`
- [ ] `day1_lab_outputs/detection_predictions.json`
- [ ] `day1_lab_outputs/segmentation_predictions.json`
- [ ] `day1_lab_outputs/IMAGE_ATTRIBUTION.md`
- [ ] `day1_lab_outputs/visuals/classification_top5.png`
- [ ] `day1_lab_outputs/visuals/detection_predictions.png`
- [ ] `day1_lab_outputs/visuals/segmentation_prediction.png`
- [ ] Ô validation cuối notebook báo `PASS`
- [ ] Không có họ tên, MSSV hoặc dữ liệu nhạy cảm trong báo cáo/output
- [ ] Đã nộp **link repository** trên LMS