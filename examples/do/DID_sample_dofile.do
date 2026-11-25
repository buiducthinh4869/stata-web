* 1. THIẾT LẬP MÔI TRƯỜNG & TẠO DỮ LIỆU BẢNG (PANEL DATA)
*----------------------------------------------------------
clear all
set more off
set seed 20251206
set scheme s2color

* 1.1. Tạo cấu trúc Panel
* Giả sử ta quan sát 1000 doanh nghiệp trong 10 năm (2011-2020)
set obs 1000
gen firm_id = _n // ID doanh nghiệp

* Mở rộng thành 10 năm cho mỗi doanh nghiệp
expand 10
bysort firm_id: gen year = 2010 + _n
xtset firm_id year // Khai báo dữ liệu mảng

* 1.2. Tạo nhóm Can thiệp (Treatment Group)
* 50% doanh nghiệp nằm ở "Vùng quy hoạch" (Treatment = 1)
* 50% nằm ngoài (Treatment = 0) - Biến này không đổi theo thời gian (Time-invariant)
gen treated = (firm_id <= 500)
label define tr_lab 0 "Nhóm Đối chứng" 1 "Nhóm Can thiệp"
label values treated tr_lab

* 1.3. Định nghĩa thời điểm can thiệp (Post)
* Chính sách bắt đầu từ năm 2016
gen post = (year >= 2016)

* 1.4. Tạo biến Outcome (Y): Lợi nhuận doanh nghiệp
* - Time Effect: Xu hướng chung kinh tế tăng trưởng mỗi năm (+5 đv).
* - Group Effect: Nhóm Can thiệp vốn dĩ có lợi nhuận thấp hơn nhóm Đối chứng (-10 đv) -> Sự khác biệt ban đầu.
* - TRUE EFFECT (ATT): Chính sách làm tăng lợi nhuận thêm 15 đơn vị (Cho nhóm Can thiệp, vào thời kỳ Post).
* - Xu hướng song song: Trước 2016, hai nhóm tăng trưởng giống nhau (hệ số của time * treated = 0).

local true_att = 15
gen revenue = 100 + (-10 * treated) + (5 * year - 10000) + /// Baseline & Time Trend
              `true_att' * (treated * post) + /// True DID Effect
              rnormal(0, 5) // Sai số ngẫu nhiên

label var revenue "Lợi nhuận doanh nghiệp"

*----------------------------------------------------------
* 2. PHÂN TÍCH TRỰC QUAN (VISUALIZATION) - BẮT BUỘC
*----------------------------------------------------------
* Bước quan trọng nhất: Vẽ biểu đồ đường để kiểm tra Parallel Trends bằng mắt.
* Nếu hai đường không song song trước năm 2016, DiD vứt đi.

* Tính trung bình lợi nhuận theo năm và theo nhóm
preserve
    collapse (mean) revenue, by(year treated)
    
    twoway (line revenue year if treated==1, lcolor(red) lwidth(thick) lpattern(solid)) ///
           (line revenue year if treated==0, lcolor(blue) lwidth(thick) lpattern(dash)) ///
           , xline(2015.5, lcolor(black) lpattern(dot)) ///
           legend(label(1 "Nhóm Can thiệp") label(2 "Nhóm Đối chứng")) ///
           title("Biểu đồ Xu hướng Lợi nhuận (Parallel Trends)") ///
           xtitle("Năm") ytitle("Lợi nhuận trung bình") ///
           xlabel(2011(1)2020) ///
           note("Dấu chấm đen: Thời điểm chính sách bắt đầu")
restore

* -> GỢI Ý GIẢNG DẠY:
* Chỉ vào đồ thị: "Các em thấy từ 2011-2015, hai đường màu đỏ và xanh đi song song nhau không? 
* Đó là bằng chứng của Parallel Trends. Sau vạch đen, đường đỏ vọt lên, đó là Tác động (ATT)."

*----------------------------------------------------------
* 3. ƯỚC LƯỢNG MÔ HÌNH DID CỔ ĐIỂN (CANONICAL DID)
*----------------------------------------------------------
* Mô hình chuẩn: Y = a + b1*Post + b2*Treated + beta*(Post*Treated) + e
* Biến tương tác (did_coef) chính là kết quả ta cần.

display "----------------------------------------------------"
display "KẾT QUẢ ƯỚC LƯỢNG DID (MÔ HÌNH 2 CHIỀU CỐ ĐỊNH)"
display "True Effect kỳ vọng = 15"
display "----------------------------------------------------"

* Sử dụng reg với biến tương tác (Factor notation ##)
* vce(cluster firm_id): Bắt buộc trong Panel Data để xử lý tự tương quan (Serial Correlation)
reg revenue i.post##i.treated, vce(cluster firm_id)

* Hoặc dùng xtreg (Mô hình FE chính thống hơn) - Kết quả tương tự
xtreg revenue i.post##i.treated i.year, fe vce(cluster firm_id)

* -> Hướng dẫn đọc:
* Nhìn dòng "1.post#1.treated". Hệ số này phải gần bằng 15.

*----------------------------------------------------------
* 4. PHÂN TÍCH EVENT STUDY (KIỂM ĐỊNH SONG SONG NÂNG CAO)
*----------------------------------------------------------
* Đây là tiêu chuẩn vàng hiện nay của các bài báo Top-tier.
* Thay vì 1 biến dummy Post, ta chia nhỏ ra từng năm để xem
* tác động động (dynamic effects) và kiểm tra giả định song song.

* Tạo các biến tương tác cho từng năm (Trừ năm cơ sở 2015 - ngay trước can thiệp)
* Ta muốn xem:
* - Trước 2015: Các hệ số có = 0 không? (Nếu = 0 -> Thỏa mãn song song)
* - Sau 2015: Các hệ số có > 0 không? (Tác động chính sách)

display "----------------------------------------------------"
display "KẾT QUẢ EVENT STUDY REGRESSION"
display "----------------------------------------------------"

* ib2015.year: Chọn năm 2015 làm năm gốc (Base category)
reg revenue i.treated##ib2015.year, vce(cluster firm_id)

* Lưu lại kết quả để vẽ hình
estimates store event_study

* Vẽ biểu đồ Coefplot (Cần cài: ssc install coefplot)
* Vì giảng đường có thể chưa cài coefplot, ta dùng cách thủ công đơn giản
* nhưng nếu có gói lệnh thì dùng dòng dưới đây sẽ rất đẹp:
coefplot event_study, keep(1.treated#*.year) vertical yline(0) drop(1.treated#2015.year)

display ">>> MẸO GIẢNG DẠY: <<<"
display "Nhìn bảng kết quả phần tương tác (1.treated # year):"
display "1. Các năm 2011, 2012, 2013, 2014: Hệ số (Coef) KHÔNG có ý nghĩa thống kê (P-value lớn)."
display "   -> Kết luận: Không có sự khác biệt trước can thiệp (No Pre-trends). Giả định OK."
display "2. Các năm 2016-2020: Hệ số dương lớn (~15) và Có ý nghĩa thống kê."
display "   -> Kết luận: Chính sách có hiệu quả bền vững."

*=============================================================================
* KẾT THÚC BÀI GIẢNG
*=============================================================================