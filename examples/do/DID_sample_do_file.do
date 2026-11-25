* =============================================================================
* PHẦN 1: PHÂN TÍCH DID VỚI DỮ LIỆU BẢNG (PANEL DATA)
* =============================================================================

use "D:\My work\Kinh tế lượng\Kinh tế lượng nâng cao\Đánh giá tác động\DID\Dữ liệu bảng.dta", clear

* Bước 1: Khai báo cấu trúc dữ liệu bảng
xtset id t

* -----------------------------------------------------------------------------
* CÁCH 1: PHƯƠNG PHÁP DID CỔ ĐIỂN DÙNG TWFE (xtreg, fe)
* (Hữu ích để giảng dạy về mặt khái niệm, nhưng có nhiều hạn chế)
* -----------------------------------------------------------------------------
* Lưu ý: Để chạy mô hình này, ta cần biến tương tác giữa nhóm can thiệp và thời gian sau can thiệp.
* Giả sử biến 'treated' của bạn là biến 0/1 cho nhóm đối chứng/can thiệp
* và 't' là biến thời gian (ví dụ 0=trước, 1=sau).
* Ta cần tạo biến tương tác. Trong Stata, có thể dùng toán tử ##

* Mô hình DID đơn giản nhất (không có biến kiểm soát):
xtreg y i.t##i.treated, fe vce(cluster id)
* Giải thích cho sinh viên:
* - i.t: Bao gồm các đặc điểm cố định theo thời gian (time fixed effects).
* - fe: Bao gồm các đặc điểm cố định của từng đối tượng (entity fixed effects).
* - Hệ số của t#treated (ví dụ 1.t#1.treated) chính là ước lượng ATET.

* Mô hình DID đầy đủ với biến kiểm soát:
xtreg y i.t##i.treated c.x1##c.x2, fe vce(cluster id)
* Chú ý: Biến 'treated' riêng lẻ sẽ bị loại bỏ như đã giải thích.

* -----------------------------------------------------------------------------
* CÁCH 2: PHƯƠNG PHÁP DID HIỆN ĐẠI (xtdidregress) - ĐƯỢC KHUYẾN KHÍCH
* (Mạnh mẽ hơn, linh hoạt hơn, đặc biệt với can thiệp so le)
* -----------------------------------------------------------------------------
* Stata sẽ tự động tạo các biến tương tác cần thiết.
* Giả sử 'treated' là biến chỉ định nhóm nào sẽ nhận can thiệp tại một thời điểm nào đó.

xtdidregress (y c.x1##c.x2) (treated), group(id) time(t)

* -----------------------------------------------------------------------------
* Bước 3: Kiểm định các giả định quan trọng (SAU KHI CHẠY xtdidregress)
* -----------------------------------------------------------------------------

* a. Kiểm tra xu hướng song song bằng đồ thị (event study plot)
* Gợi ý: Sửa 'trendplot' thành 'trendplots' (có chữ 's')
estat trendplots

* b. Kiểm định giả định xu hướng song song (Parallel Trends Assumption)
* Kiểm định F chung cho các hệ số của thời gian trước can thiệp (pre-treatment periods).
* H0: Các hệ số trước can thiệp đồng thời bằng 0 (tức là xu hướng song song).
estat ptrend
* Diễn giải: Nếu p-value > 0.1, ta không thể bác bỏ H0, ủng hộ giả định xu hướng song song.

* c. Kiểm tra xu hướng có thể dự đoán trước (Granger Causality-type test)
* H0: Kết quả (y) trong quá khứ không dự đoán được việc sẽ nhận can thiệp trong tương lai.
estat granger
* Diễn giải: Nếu p-value > 0.1, ta không thể bác bỏ H0. Đây là một dấu hiệu tốt,
* cho thấy việc lựa chọn nhóm can thiệp không dựa trên xu hướng trước đó của biến kết quả.




* =============================================================================
* PHẦN 2: PHÂN TÍCH DID VỚI DỮ LIỆU CHÉO LẶP LẠI
* =============================================================================

use "D:\My work\Kinh tế lượng\Kinh tế lượng nâng cao\Đánh giác tác động\DID\Dữ liệu chéo.dta", clear

* -----------------------------------------------------------------------------
* CÁCH 1: PHƯƠNG PHÁP DID CỔ ĐIỂN DÙNG OLS (reg)
* (Rất tốt để dạy khái niệm DID cơ bản nhất)
* -----------------------------------------------------------------------------
* Ta cần tạo thủ công 3 biến chính:
* 1. Biến 'treated': =1 nếu thuộc nhóm can thiệp (vd: hospital = A, B), =0 nếu thuộc nhóm đối chứng (hospital = C, D)
* 2. Biến 'post': =1 nếu là thời gian sau can thiệp (vd: month >= 6), =0 nếu là trước đó.
* 3. Biến tương tác 'did': did = treated * post

* Ví dụ tạo biến (bạn cần điều chỉnh theo dữ liệu thực tế)
* Giả sử can thiệp diễn ra từ tháng 6 và chỉ ở bệnh viện 1.
gen treated = (hospital == 1)
gen post = (month >= 6) & (month != .)
gen did_interaction = treated * post

* Chạy mô hình hồi quy DID cổ điển
* satis = B0 + B1*treated + B2*post + B3*did_interaction + e
reg satis i.treated i.post i.did_interaction, vce(cluster hospital)
* Diễn giải: Hệ số của `1.did_interaction` chính là ước lượng ATET.

* -----------------------------------------------------------------------------
* CÁCH 2: PHƯƠNG PHÁP DID HIỆN ĐẠI (didregress) - ĐƯỢC KHUYẾN KHÍCH
* -----------------------------------------------------------------------------
* Giả sử biến `procedure` của bạn là biến chỉ định can thiệp (ví dụ: =1 nếu hospital đó được can thiệp TẠI thời điểm month đó, =0 nếu ngược lại).
* Nếu `procedure` chỉ là biến phân nhóm (luôn = 1 cho bệnh viện được can thiệp), thì cú pháp của bạn đúng.

didregress (satis) (procedure), group(hospital) time(month) vce(cluster hospital)
* Ghi chú: `vce(cluster hospital)` thường được `didregress` tự động xử lý khi có `group(hospital)` nhưng ghi rõ ra cũng không sao.

* -----------------------------------------------------------------------------
* Bước 3: Kiểm định giả định và tóm tắt (SAU KHI CHẠY didregress)
* -----------------------------------------------------------------------------
* Đồ thị xu hướng song song
estat trendplots

* Kiểm định xu hướng song song (p-value > 0.05 ==> xu hướng song song)
estat ptrend 
// Kiểm định H0: Xu hướng của hai nhóm là song song trước khi can thiệp.

* Kiểm định xu hướng dự đoán (p-value > 0.1 ==> tốt)
estat granger 
// Kiểm định H0: Việc nhận can thiệp không bị ảnh hưởng bởi xu hướng trước đó của biến kết quả.

* Tóm tắt cấu trúc dữ liệu, số quan sát mỗi nhóm và thời gian
estat summarize