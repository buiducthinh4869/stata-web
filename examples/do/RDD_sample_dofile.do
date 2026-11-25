* BƯỚC 1: THIẾT LẬP MÔI TRƯỜNG VÀ MÔ PHỎNG SỐ LIỆU (DGP)
*----------------------------------------------------------
clear all
set more off
set seed 20251206
* Cài đặt các gói lệnh cần thiết (nếu chưa có)
* ssc install rdrobust, replace
* ssc install rddensity, replace

* 1.1. Tạo mẫu
set obs 5000

* 1.2. Tạo biến "chạy" (Running Variable)
* Điểm thi đầu vào, giả định phân phối đều từ 0 đến 100
gen score_entry = runiform() * 100
label var score_entry "Điểm thi đầu vào (Running Variable)"

* 1.3. Định nghĩa ngưỡng (Cutoff) và biến can thiệp (Treatment)
* Đây là thiết kế "Sharp RDD": Ai >= 50 là chắc chắn nhận học bổng.
local cutoff = 50
gen scholarship = (score_entry >= `cutoff')
label define schol_lab 0 "Không nhận HB" 1 "Nhận HB"
label values scholarship schol_lab
label var scholarship "Nhận học bổng (Treatment)"

* 1.4. Tạo biến kết quả (Outcome Variable)
* Giả định: Điểm cuối khóa phụ thuộc vào điểm đầu vào (một cách phi tuyến)
* và được TĂNG THÊM 8 ĐIỂM nếu nhận học bổng (đây là True Effect).
* τ (Tau) = 8
local true_effect = 8

* Mô hình tạo dữ liệu (DGP)
gen score_final = 10 + 0.8 * score_entry - 0.005 * score_entry^2 + ///
                  `true_effect' * scholarship + rnormal(0, 5)

* Chuẩn hóa biến chạy quanh ngưỡng 0 (thực hành tốt nhất)
gen score_centered = score_entry - `cutoff'
label var score_centered "Điểm thi đầu vào (Chuẩn hóa tại ngưỡng 0)"

label var score_final "Điểm thi cuối khóa (Outcome)"
format %9.2f score*


*----------------------------------------------------------
* BƯỚC 2: PHÂN TÍCH TRỰC QUAN - "EYEBALL TEST" (CỰC KỲ QUAN TRỌNG)
*----------------------------------------------------------
* Một bài báo RDD tốt luôn bắt đầu bằng đồ thị trực quan.
* Đồ thị này phải cho thấy một "bước nhảy" (jump) rõ ràng tại ngưỡng.

rdplot score_final score_centered, c(`=0`) ///
    p(1) /// Bậc đa thức (1=tuyến tính)
    binselect(esmv) /// Phương pháp chọn bin tự động
    kernel(triangular) /// Dạng hạt nhân
    title("Quan hệ giữa Điểm đầu vào và Điểm cuối khóa", size(medium)) ///
    xtitle("Điểm đầu vào (chuẩn hóa tại ngưỡng 0)", size(small)) ///
    ytitle("Điểm cuối khóa", size(small)) ///
    legend(off) ///
    graph_options(note("Đồ thị cho thấy một bước nhảy dương tại ngưỡng 0," ///
    "gợi ý tác động tích cực của học bổng."))

* -> Mẹo giảng dạy: Hãy giải thích cho nghiên cứu sinh rằng nếu không có bước nhảy này,
* phương pháp RDD không đáng tin cậy. Đây là bằng chứng đầu tiên.


*----------------------------------------------------------
* BƯỚC 3: ƯỚC LƯỢNG THAM SỐ - PHƯƠNG PHÁP HIỆN ĐẠI
*----------------------------------------------------------
* Chuẩn mực quốc tế hiện nay là sử dụng hồi quy nhân cục bộ (local kernel regression)
* với bandwidth tối ưu của Calonico, Cattaneo, and Titiunik (CCT).
* Lệnh 'rdrobust' trong Stata thực hiện chính xác điều này.

display "------------------------------------------------"
display "KẾT QUẢ ƯỚC LƯỢNG RDD TỐI ƯU (CCT, 2014)"
display "------------------------------------------------"
rdrobust score_final score_centered

* -> Hướng dẫn đọc kết quả cho sinh viên:
* 1. Nhìn vào "Conventional": Đây là ước lượng thông thường.
* 2. Nhìn vào "Robust": Đây là ước lượng đã được hiệu chỉnh sai số (bias-corrected),
*    đáng tin cậy hơn và là con số cần báo cáo chính.
* 3. Coefficient ~ 8, rất gần với `true_effect` mà ta đã lập trình.


*----------------------------------------------------------
* BƯỚC 4: KIỂM ĐỊNH TÍNH VỮNG (ROBUSTNESS & VALIDITY CHECKS)
*----------------------------------------------------------
* Một bài báo RDD xuất sắc phải có các kiểm định này để tăng tính thuyết phục.

* 4.1. Kiểm định McCrary (2008): Kiểm tra sự thao túng tại ngưỡng
* Ý tưởng: Nếu sinh viên/giảng viên có thể "cố" để vượt qua ngưỡng 50,
* mật độ phân phối của biến chạy sẽ có một bước nhảy bất thường tại ngưỡng.
* Null Hypothesis: Không có sự thao túng (phân phối liên tục).
display "------------------------------------------------"
display "KIỂM ĐỊNH MCCRARY VỀ THAO TÚNG NGƯỠNG"
display "------------------------------------------------"
rddensity score_entry, c(`cutoff')
* -> Kết quả cần tìm: T-statistic không có ý nghĩa thống kê (p-value lớn).
*    Đồ thị mật độ cũng phải trông "mượt" qua ngưỡng.

* 4.2. Kiểm định Giả dược (Placebo Test)
* Ý tưởng: Thử chạy mô hình tại một ngưỡng "giả" mà ở đó không có can thiệp nào xảy ra.
* Kết quả ước lượng tại ngưỡng giả này phải bằng 0.
display "------------------------------------------------"
display "KIỂM ĐỊNH GIẢ DƯỢC TẠI NGƯỠNG 30 VÀ 70"
display "------------------------------------------------"
* Kiểm tra tại ngưỡng = 30
rdrobust score_final score_entry, c(30)
* Kiểm tra tại ngưỡng = 70
rdrobust score_final score_entry, c(70)
* -> Kết quả mong đợi: Hệ số ước lượng gần 0 và không có ý nghĩa thống kê.

* 4.3. Kiểm tra sự cân bằng của các biến đồng hành (Covariate Balance)
* Ý tưởng: Tương tự RCT, các đặc điểm có từ trước của sinh viên (vd: giới tính, tuổi)
* phải liên tục qua ngưỡng. Nếu học bổng chỉ dành cho nữ sinh viên có điểm >50,
* thì sẽ có bước nhảy ở biến "nữ", vi phạm giả định RDD.
* Tạo một biến giả định:
gen male = rbinomial(1, 0.5)
label var male "Giới tính Nam (1=Nam, 0=Nữ)"

display "------------------------------------------------"
display "KIỂM TRA CÂN BẰNG BIẾN ĐỒNG HÀNH (GIỚI TÍNH)"
display "------------------------------------------------"
rdrobust male score_centered
* -> Kết quả mong đợi: Không tìm thấy bước nhảy có ý nghĩa thống kê ở biến 'male'.