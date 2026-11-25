*----------------------------------------------------------
* BƯỚC 1: THIẾT LẬP MÔI TRƯỜNG VÀ MÔ PHỎNG SỐ LIỆU (DGP)
*----------------------------------------------------------
clear all
set more off
set seed 20241206 // Đặt seed để kết quả giống nhau mỗi lần chạy (cho giảng dạy)

* 1.1 Tạo mẫu nghiên cứu
set obs 2000 // Giả lập 2000 cá nhân

* 1.2 Tạo các biến đặc điểm nền (Covariates - X)
* Những biến này tồn tại TRƯỚC khi có can thiệp
gen age = round(rnormal(35, 5))         // Tuổi: Trung bình 35, độ lệch 5
gen experience = round(rnormal(10, 3))  // Kinh nghiệm: Trung bình 10 năm
gen education = round(rnormal(12, 2))   // Số năm đi học
gen ability = rnormal(0, 1)             // Năng lực (biến ẩn, không quan sát được)

* Đảm bảo logic dữ liệu (không có tuổi hoặc kinh nghiệm âm)
replace experience = 0 if experience < 0

* 1.3 Gán ngẫu nhiên vào nhóm Can thiệp (Treatment Assignment)
* Đây là cốt lõi của RCT: Việc nhận treatment hoàn toàn ngẫu nhiên (tung đồng xu)
gen treatment = rbinomial(1, 0.5) 

label define treat_lab 0 "Control (Đối chứng)" 1 "Treatment (Can thiệp)"
label values treatment treat_lab

* 1.4 Tạo biến kết quả (Outcome - Y)
* Giả định: 
* - Thu nhập cơ bản phụ thuộc vào: Tuổi, Kinh nghiệm, Giáo dục và Năng lực.
* - Hiệu quả THỰC SỰ của chương trình (True Treatment Effect) là tăng 500$ thu nhập.
* - Sai số ngẫu nhiên (Error term).

local true_effect = 500
gen error = rnormal(0, 300)

* Công thức tạo Y (Data Generation Process):
gen income = 1000 + (20 * age) + (50 * experience) + (100 * education) + (200 * ability) + (`true_effect' * treatment) + error

format income %9.0f
label var income "Thu nhập sau chương trình ($)"

*----------------------------------------------------------
* BƯỚC 2: KIỂM TRA SỰ CÂN BẰNG (BALANCE CHECK)
*----------------------------------------------------------
* Mục tiêu: Chứng minh ngẫu nhiên hóa thành công.
* Nhóm Treatment và Control phải tương đồng về đặc điểm nền trước can thiệp.

display "------------------------------------------------"
display "KIỂM ĐỊNH T-TEST VỀ SỰ CÂN BẰNG CÁC ĐẶC ĐIỂM NỀN"
display "------------------------------------------------"

* Cách thủ công (để SV hiểu bản chất)
ttest age, by(treatment)
ttest experience, by(treatment)
ttest education, by(treatment)

* Mẹo giảng dạy: Hãy chỉ cho SV thấy 'Ha: diff != 0' có Pr(|T| > |t|) > 0.05
* Nghĩa là KHÔNG có sự khác biệt có ý nghĩa thống kê giữa 2 nhóm.

* (Nâng cao: Nếu bạn dùng lệnh người dùng viết 'iebaltab' hoặc 'orth_out' sẽ đẹp hơn, nhưng code gốc này chạy được trên mọi máy).

*----------------------------------------------------------
* BƯỚC 3: ĐÁNH GIÁ TÁC ĐỘNG (ESTIMATION)
*----------------------------------------------------------
display "------------------------------------------------"
display "KẾT QUẢ HỒI QUY ĐÁNH GIÁ TÁC ĐỘNG"
display "------------------------------------------------"

* Cách 1: So sánh trung bình đơn giản (Simple Difference in Means)
* Vì là RCT, E[Y1 - Y0] chính là hiệu quả can thiệp.
reg income treatment
* -> Hãy yêu cầu SV so sánh hệ số treatment với con số 500 (True effect) thiết lập ban đầu.

* Cách 2: Hồi quy có kiểm soát biến (Covariate Adjustment)
* Mục tiêu: Tăng độ chính xác (giảm Standard Error), không làm thay đổi nhiều hệ số beta.
reg income treatment age experience education

*----------------------------------------------------------
* BƯỚC 4: TRỰC QUAN HÓA (OPTIONAL)
*----------------------------------------------------------
twoway (kdensity income if treatment==0, color(red%30) lcolor(red)) ///
       (kdensity income if treatment==1, color(blue%30) lcolor(blue)), ///
       legend(label(1 "Control Group") label(2 "Treatment Group")) ///
       title("Phân phối thu nhập giữa 2 nhóm") ///
       note("Nhóm Treatment dịch chuyển sang phải so với Control")