* 1. THIẾT LẬP MÔI TRƯỜNG
*----------------------------------------------------------
clear all
set more off
set seed 20251206   // Cố định kết quả để sinh viên ra giống thầy
set scheme s2color  // Giao diện sáng sủa cho đồ thị

* Cài đặt gói lệnh psmatch2 (Chạy 1 lần nếu máy chưa có)
* ssc install psmatch2, replace

* 2. MÔ PHỎNG SỐ LIỆU (DGP)
*----------------------------------------------------------
set obs 5000

* 2.1. Tạo biến độc lập (Covariates)
* Tuổi (Age)
gen age = rnormal(40, 10)
* Động lực (Motivation) - Biến quan trọng gây thiên lệch
gen motivation = runiform(0, 10)
* Học vấn (Educ) - 1: Có bằng ĐH, 0: Không
gen educ = rbinomial(1, 0.4)

* 2.2. Tạo cơ chế tự lựa chọn (Selection Bias)
* Người có động lực cao, học vấn cao -> Xác suất đi học (Training) cao hơn
gen z = -6 + 0.1*age + 0.9*motivation + 1.2*educ + rnormal(0, 1)
gen prob_score = exp(z) / (1 + exp(z)) // Đây chính là Propensity Score thực
gen training = rbinomial(1, prob_score)

label define tr_lab 0 "Đối chứng (Control)" 1 "Can thiệp (Treated)"
label values training tr_lab

* 2.3. Tạo biến kết quả (Outcome: Thu nhập)
* True Effect (Tác động thực) = 500 USD
* Lưu ý: Motivation cũng làm tăng thu nhập -> Gây nhiễu nếu không xử lý
gen income = 1200 + 20*age + 200*motivation + 300*educ + ///
             500*training + rnormal(0, 300)

label var training "Tham gia đào tạo"
label var income "Thu nhập sau khóa học"

*----------------------------------------------------------
* 3. THỰC HIỆN PSM VÀ VẼ BIỂU ĐỒ COMMON SUPPORT
*----------------------------------------------------------

* Bước 3.1: Ước lượng PSM
* Sử dụng thuật toán láng giềng gần nhất (Nearest Neighbor 1-to-1)
* outcome(income): Biến kết quả
* common: Chỉ giữ lại các quan sát nằm trong miền giá trị chung
display ">>> KẾT QUẢ PSM <<<"
psmatch2 training age motivation educ, outcome(income) neighbor(1) common logit
teffects psmatch (income) (training age motivation educ), atet
*----------------------------------------------------------
* Bước 3.2: VẼ BIỂU ĐỒ COMMON SUPPORT (CÁCH 1 - CHUẨN)
*----------------------------------------------------------
* Lệnh 'psgraph' đi kèm với psmatch2
* Nó vẽ hai biểu đồ cột (histogram) đối đầu nhau để xem độ chồng lấn.

psgraph, bin(20) ///
    title("Biểu đồ Miền giá trị chung (Common Support)", size(medium)) ///
    xtitle("Điểm xu hướng (Propensity Score)") ///
    ytitle("Mật độ phân phối") ///
    note("Nhóm được điều trị (Trên) vs Nhóm đối chứng (Dưới)")

* -> GIẢI THÍCH CHO SINH VIÊN:
* Nhìn vào đồ thị: Nếu các cột của nhóm trên và nhóm dưới "phủ" lên nhau
* (overlap) càng nhiều thì khả năng tìm được cặp ghép càng tốt.
* Nếu cột lệch hẳn nhau (ví dụ: Treated dồn hết về bên phải, Control về bên trái)
* thì vi phạm giả định Common Support.

*----------------------------------------------------------
* Bước 3.3: VẼ BIỂU ĐỒ COMMON SUPPORT (CÁCH 2 - NÂNG CAO & ĐẸP HƠN)
*----------------------------------------------------------
* Cách này dùng kdensity vẽ hai đường cong mượt mà đè lên nhau.
* _pscore là biến chứa điểm xu hướng do psmatch2 tạo ra.

twoway (kdensity _pscore if training==1, color(red%30) lcolor(red) lwidth(medium)) ///
       (kdensity _pscore if training==0, color(blue%30) lcolor(blue) lwidth(medium)), ///
       title("Mật độ điểm xu hướng (Propensity Score Density)") ///
       xtitle("Xác suất tham gia (Propensity Score)") ///
       ytitle("Mật độ") ///
       legend(label(1 "Nhóm Can thiệp") label(2 "Nhóm Đối chứng")) ///
       graphregion(color(white))


*----------------------------------------------------------
* 4. KIỂM TRA CÂN BẰNG BIẾN & KẾT QUẢ (PHỤ TRỢ)
*----------------------------------------------------------
* Kiểm tra xem sau khi ghép, đặc điểm 2 nhóm đã giống nhau chưa
pstest age motivation educ, both graph

* Xem kết quả tác động (ATT)
display "Tác động thực (True Effect) là 500."
display "Tác động ước lượng (ATT) là:"
display r(att)

*----------------------------------------------------------
* 5. MỞ RỘNG: CÁC TIÊU CHUẨN MATCHING KHÁC (ALGORITHMS)
*----------------------------------------------------------
* Để sinh viên hiểu rõ hơn, ta sẽ chạy thử các thuật toán khác nhau 
* và so sánh với True Effect (500 USD).

* Lưu ý: Mỗi lần chạy psmatch2, nó sẽ tạo ra các biến _weight, _id, _n1...
* Nên ta không cần xóa thủ công, nó sẽ tự đè lên (overwrite).

*==========================================================
* 5.1. k-Nearest Neighbors (k-NN) - Lấy k láng giềng
*==========================================================
* Thay vì lấy 1 người, ta lấy 5 người giống nhất để so sánh.
* Ưu điểm: Giảm phương sai (kết quả ổn định hơn).
* Nhược điểm: Có thể tăng thiên lệch nếu 5 người đó không thực sự giống hệt.

display ">>> KẾT QUẢ: 5-NEAREST NEIGHBORS <<<"
psmatch2 training age motivation educ, outcome(income) neighbor(5) common
* Lưu lại kết quả để so sánh sau
scalar att_nn5 = r(att)

*==========================================================
* 5.2. Caliper Matching (Bán kính)
*==========================================================
* Chỉ ghép cặp nếu khoảng cách điểm xu hướng < mức cho phép (ví dụ 0.05).
* Nếu người được trị liệu (Treated) mà tìm người đối chứng (Control) 
* cách xa quá 0.05 -> Hủy ghép cặp đó.
* Ưu điểm: Đảm bảo chất lượng ghép cặp rất cao (giảm thiên lệch).
* Nhược điểm: Có thể bị mất mẫu nhiều (nếu đặt caliper quá nhỏ).

display ">>> KẾT QUẢ: CALIPER MATCHING (0.05) <<<"
psmatch2 training age motivation educ, outcome(income) neighbor(1) caliper(0.05) common
scalar att_caliper = r(att)

*==========================================================
* 5.3. Kernel Matching (Hạt nhân)
*==========================================================
* Không ghép 1-1 hay 1-5, mà dùng trung bình trọng số của TOÀN BỘ nhóm Control.
* Người nào càng giống (khoảng cách propensity score gần) thì trọng số càng lớn.
* Ưu điểm: Tận dụng tối đa dữ liệu (Efficiency cao).
* Nhược điểm: Thuật toán phức tạp, khó giải thích trực quan từng cặp.

display ">>> KẾT QUẢ: KERNEL MATCHING <<<"
psmatch2 training age motivation educ, outcome(income) kernel common
scalar att_kernel = r(att)

*----------------------------------------------------------
* 6. TỔNG HỢP SO SÁNH CÁC PHƯƠNG PHÁP
*----------------------------------------------------------
display _newline(5) // Xuống dòng cho dễ nhìn
display "=================================================="
display "       BẢNG SO SÁNH KẾT QUẢ CÁC PHƯƠNG PHÁP       "
display "=================================================="
display "Tác động thực (TRUE EFFECT)     : 500"
display "--------------------------------------------------"
display "1. Nearest Neighbor (1-to-1)    : " att_nn5    // Lưu ý: Đây là ví dụ, code trên đè biến nên lấy kq gần nhất, chỉnh lại ở dưới
* Để so sánh chính xác trong code tự động, cần lưu locals ngay sau mỗi lệnh.
* Dưới đây tôi chạy lại nhanh và in ra màn hình luôn để bạn giảng bài:

* Chạy lại NN(1) để lấy số liệu
quietly psmatch2 training age motivation educ, outcome(income) neighbor(1) common
display "1. NN (1-to-1) ATT              : " %9.2f r(att) " (Sai số chuẩn thấp, dễ trùng lặp thấp)"

* Chạy lại NN(5)
quietly psmatch2 training age motivation educ, outcome(income) neighbor(5) common
display "2. NN (5 Neighbors) ATT         : " %9.2f r(att) " (Giảm phương sai, mẫu số liệu dùng nhiều hơn)"

* Chạy lại Caliper
quietly psmatch2 training age motivation educ, outcome(income) neighbor(1) caliper(0.05) common
display "3. Radius/Caliper (0.05) ATT    : " %9.2f r(att) " (Loại bỏ các cặp kém chất lượng)"

* Chạy lại Kernel
quietly psmatch2 training age motivation educ, outcome(income) kernel common
display "4. Kernel Matching ATT          : " %9.2f r(att) " (Dùng toàn bộ đối chứng có trọng số)"

display "=================================================="
display "LỜI KHUYÊN CHO SINH VIÊN:"
display "- Nếu cỡ mẫu nhỏ: Dùng Kernel hoặc NN(5) để tận dụng dữ liệu."
display "- Nếu dữ liệu 'rộng' (Common support tốt): NN(1) hoặc Caliper là đủ tốt và dễ bảo vệ."
display "=================================================="

* Xóa các biến tạm do psmatch2 tạo ra để dữ liệu sạch đẹp
cap drop _pscore _treated _support _weight _income _id _n1 _nn _pdif


kmatch ps training age motivation educ (income), att
kmatch csummarize
kmatch cdensity
kmatch ccumul
kmatch cbox

*----------------------------------------------------------
* KẾT THÚC DO-FILE
*----------------------------------------------------------