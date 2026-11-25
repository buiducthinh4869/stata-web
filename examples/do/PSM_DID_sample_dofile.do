* ==============================================================================
* BÀI GIẢNG: PSM-DID (PROPENSITY SCORE MATCHING DÙNG CHO DỮ LIỆU BẢNG)
* Cập nhật: Sửa lỗi "outcome does not vary"
* ==============================================================================

clear all
set seed 2025 // Cố định seed để kết quả mô phỏng giống nhau giữa các học viên

* ------------------------------------------------------------------------------
* BƯỚC 1: LOAD DỮ LIỆU & CHUẨN BỊ
* ------------------------------------------------------------------------------
* Đường dẫn dữ liệu của bạn
use "D:\My work\Kinh tế lượng\Kinh tế lượng nâng cao\Đánh giá tác động\DID\Dữ liệu bảng.dta", clear

* Khai báo cấu trúc dữ liệu bảng
xtset id t

* --- KHẮC PHỤC LỖI R(2000): TẠO BIẾN ĐỊNH DANH NHÓM (Time-invariant Group) ---
* Vấn đề: Biến 'treated' hiện tại thay đổi theo thời gian (0 trước, 1 sau).
* PSM cần biết ai THUỘC nhóm can thiệp ngay cả khi họ chưa bị can thiệp ở t=1.
* Lệnh dưới đây gán treat_group = 1 cho tất cả các năm nếu ID đó từng có treated=1.
bysort id: egen treat_group = max(treated)

label variable treat_group "Nhóm Can thiệp (Cố định - Dùng cho Matching)"
label variable treated "Trạng thái Can thiệp (Thay đổi - Dùng cho DID)"

* Kiểm tra: treat_group phải cố định theo thời gian, treated thay đổi
list id t treat_group treated in 1/20, sepby(id)

* ------------------------------------------------------------------------------
* BƯỚC 2: MÔ PHỎNG BIẾN ĐẶC ĐIỂM (COVARIATES) CHO NCS THỰC HÀNH
* ------------------------------------------------------------------------------
* Giả định tình huống giảng dạy:
* Nhóm Can thiệp (treat_group=1) thường là người lớn tuổi hơn và học vấn thấp hơn
* (Đây là nguồn gốc gây ra Selection Bias cần loại bỏ bằng PSM).

* Biến Age (Tuổi tại thời điểm t=1): Lớn tuổi hơn có xu hướng vào nhóm Can thiệp cao hơn
gen noise1 = rnormal(0, 5)
by id: gen age_baseline = 35 + 8*treat_group + noise1 if _n==1
by id: replace age_baseline = age_baseline[1] // Gán cho các năm sau

* Biến Educ (Số năm đi học): Học vấn thấp hơn có xu hướng vào nhóm Can thiệp cao hơn
gen noise2 = rnormal(0, 2)
by id: gen educ_baseline = 14 - 3*treat_group + noise2 if _n==1
by id: replace educ_baseline = educ_baseline[1] 

label variable age_baseline "Độ tuổi tại t=1 (Biến matching)"
label variable educ_baseline "Năm đi học tại t=1 (Biến matching)"

* ------------------------------------------------------------------------------
* BƯỚC 3: TÍNH ĐIỂM XU HƯỚNG (PROPENSITY SCORE)
* ------------------------------------------------------------------------------
* Quan trọng: Chỉ match dựa trên dữ liệu TRƯỚC CAN THIỆP (t=1).
preserve // Lưu dữ liệu hiện tại vào bộ nhớ tạm
keep if t == 1

* Chạy Logit xác định xác suất tham gia nhóm (Dùng treat_group)
* Lưu ý: Ta dùng 'treat_group' (biến tĩnh) thay vì 'treated' (biến động = 0 ở t=1)
logit treat_group age_baseline educ_baseline x1 x2

* Dự báo điểm xu hướng
predict pscore, pr

* (Optional) Giảng dạy: Vẽ đồ thị kiểm tra vùng hỗ trợ chung (Common Support)
* twoway (kdensity pscore if treat_group==1, color(red)) (kdensity pscore if treat_group==0, color(blue)), legend(label(1 "Can thiệp") label(2 "Đối chứng")) title("Kiểm tra vùng hỗ trợ chung")

* ------------------------------------------------------------------------------
* BƯỚC 4: TẠO TRỌNG SỐ (INVERSE PROBABILITY WEIGHTING - IPW)
* ------------------------------------------------------------------------------
* Ta sử dụng trọng số để tái cấu trúc mẫu sao cho giống nhóm Can thiệp nhất (ATT)

gen att_weight = .
* Với nhóm can thiệp: Giữ nguyên (trọng số = 1)
replace att_weight = 1 if treat_group == 1
* Với nhóm đối chứng: Gán trọng số dựa trên điểm pscore
replace att_weight = pscore / (1 - pscore) if treat_group == 0

* Giữ lại ID và trọng số để ghép về
keep id att_weight pscore
tempfile matching_data
save `matching_data'

restore // Khôi phục lại dữ liệu bảng đầy đủ

* Ghép trọng số vừa tính vào dữ liệu bảng
merge m:1 id using `matching_data'
keep if _merge == 3
drop _merge

* ------------------------------------------------------------------------------
* BƯỚC 5: ƯỚC LƯỢNG DID TRUYỀN THỐNG VÀ PSM-DID
* ------------------------------------------------------------------------------

* --- Mô hình 1: Standard FE-DID (Chưa điều chỉnh selection bias) ---
display "--- KẾT QUẢ DID CHUẨN (TWO-WAY FIXED EFFECTS) ---"
xtreg y i.treated i.t, fe vce(cluster id)
* Ghi chú: i.treated chính là D_it (tác động can thiệp)

* --- Mô hình 2: PSM-DID (Sử dụng trọng số) ---
* Những người ở nhóm đối chứng mà có đặc điểm (tuổi, học vấn) KHÁC XA nhóm can thiệp
* sẽ có Pscore thấp => Trọng số att_weight rất nhỏ => Ít ảnh hưởng đến kết quả hồi quy.
display "--- KẾT QUẢ PSM-DID (CÓ TRỌNG SỐ IPW) ---"
xtreg y i.treated i.t [pweight=att_weight], fe vce(cluster id)

* Giải thích cho NCS:
* - So sánh hệ số của biến 1.treated ở hai mô hình.
* - Sự chênh lệch hệ số chính là phần chệch do tự lựa chọn (selection bias) 
*   dựa trên các biến quan sát được (tuổi, học vấn) gây ra.



* ==============================================================================
* BƯỚC 6: TRỰC QUAN HÓA DỮ LIỆU - CHỨNG MINH TÍNH HIỆU QUẢ CỦA PSM-DID
* (Chạy tiếp theo đoạn code trước, sau khi đã có biến 'att_weight')
* ==============================================================================

* Cài đặt giao diện đồ thị cho đẹp (nếu muốn)
graph set window fontface "Arial"

* ------------------------------------------------------------------------------
* BIỂU ĐỒ 1: COMMON SUPPORT (VÙNG HỖ TRỢ CHUNG)
* Mục đích: Cho NCS thấy trước khi match, 2 nhóm có sự phân bổ khác nhau.
* ------------------------------------------------------------------------------
twoway (kdensity pscore if treat_group==1, color(red%30) lcolor(red) lwidth(medium)) ///
       (kdensity pscore if treat_group==0, color(blue%30) lcolor(blue) lwidth(medium)), ///
       title("1. Phân phối điểm xu hướng (Common Support)", size(medium)) ///
       subtitle("Trước khi ghép cặp/trọng số") ///
       legend(label(1 "Nhóm Can thiệp") label(2 "Nhóm Đối chứng") pos(6) rows(1)) ///
       xtitle("Propensity Score") ytitle("Density") ///
       note("Lưu ý: Hai nhóm lệch nhau cho thấy cần phải thực hiện PSM") 


* ------------------------------------------------------------------------------
* BIỂU ĐỒ 2: KIỂM TRA CÂN BẰNG ĐẶC ĐIỂM (BEFORE vs AFTER WEIGHTING)
* Mục đích: Chứng minh hiệu quả kỹ thuật - Trọng số đã xóa bỏ sự khác biệt Age/Educ
* ------------------------------------------------------------------------------
preserve
    * Tính trung bình chưa cân bằng (Unweighted)
    collapse (mean) age_mean=age_baseline educ_mean=educ_baseline, by(treat_group)
    gen type = "1. Chưa cân bằng (Original)"
    tempfile unweighted_means
    save `unweighted_means'
restore

preserve
    * Tính trung bình đã cân bằng (Weighted by PSM)
    * Sử dụng aweight (analytical weight) để mô phỏng tác động của pweight trong thống kê mô tả
    collapse (mean) age_mean=age_baseline educ_mean=educ_baseline [aw=att_weight], by(treat_group)
    gen type = "2. Đã cân bằng (PSM Weighted)"
    append using `unweighted_means'
    
    * Vẽ đồ thị so sánh Age
    graph bar age_mean, over(treat_group, relabel(1 "Đối chứng" 2 "Can thiệp")) ///
        over(type) ///
        title("2. Hiệu quả cân bằng đặc điểm: Độ tuổi (Age)", size(medium)) ///
        blabel(bar, format(%9.1f)) ///
        ytitle("Độ tuổi trung bình") ///
        note("Quan sát: Ở phần 'Đã cân bằng', cột Đối chứng cao lên ngang bằng cột Can thiệp")
restore


* ------------------------------------------------------------------------------
* BIỂU ĐỒ 3: XU HƯỚNG KẾT QUẢ (OUTCOME TRENDS) - QUAN TRỌNG NHẤT
* Mục đích: So sánh 3 đường: Treated, Control (Gốc), Control (PSM).
* Để xem đường Control (PSM) có song song/tiệm cận tốt hơn không.
* ------------------------------------------------------------------------------
preserve
    * 1. Tính giá trị trung bình Y theo thời gian cho từng nhóm
    * Nhóm Can thiệp
    collapse (mean) y_treated=y if treat_group==1, by(t)
    save "temp_treated.dta", replace
    
    * 2. Nhóm Đối chứng - Gốc (Unmatched)
    restore 
    preserve
    collapse (mean) y_control_raw=y if treat_group==0, by(t)
    save "temp_control_raw.dta", replace
    
    * 3. Nhóm Đối chứng - Có trọng số (PSM Weighted Control)
    * Đây là Counterfactual (giả tưởng) được tạo ra từ PSM
    restore
    preserve
    collapse (mean) y_control_psm=y [aw=att_weight] if treat_group==0, by(t)
    
    * Gộp các file lại để vẽ
    merge 1:1 t using "temp_treated.dta", nogenerate
    merge 1:1 t using "temp_control_raw.dta", nogenerate
    
    * Xác định thời điểm can thiệp để vẽ đường dọc (t=6)
    local intervention_time = 6 
    
    * Vẽ đồ thị
    twoway (line y_treated t, lcolor(red) lwidth(thick) lpattern(solid)) ///
           (line y_control_raw t, lcolor(gray) lwidth(medium) lpattern(dash)) ///
           (line y_control_psm t, lcolor(blue) lwidth(thick) lpattern(solid)) ///
           , ///
           xline(`intervention_time', lpattern(dot) lcolor(black)) ///
           legend(order(1 "Nhóm Can thiệp" 2 "Đối chứng (Gốc - Bị lệch)" 3 "Đối chứng (PSM - Đã chỉnh)") pos(6)) ///
           title("3. So sánh xu hướng: DID thường vs. PSM-DID", size(medium)) ///
           subtitle("Đường màu Xanh (PSM) mô phỏng tốt hơn đường màu Xám") ///
           xtitle("Thời gian") ytitle("Biến kết quả (Y)") ///
           xlabel(1(1)10) ///
           text(7.5 6 "Thời điểm can thiệp", place(e))
    
    * Xóa file tạm
    erase "temp_treated.dta"
    erase "temp_control_raw.dta"
restore

* Hiển thị cùng lúc để so sánh (nếu dùng màn hình lớn giảng dạy)
* graph combine Graph_CommonSupport Graph_Balance_Age Graph_PSM_DID_Trends, cols(3) xsize(20) ysize(7)