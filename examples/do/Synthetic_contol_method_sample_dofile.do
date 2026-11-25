* 1. THIẾT LẬP VÀ CÀI ĐẶT
*----------------------------------------------------------
clear all
set more off
set seed 20251206
set scheme s2color

*----------------------------------------------------------
* CÀI ĐẶT CÁC GÓI CẦN THIẾT (CHỈ CẦN CHẠY 1 LẦN)
*----------------------------------------------------------
* SCM rất nặng về tính toán ma trận và vẽ hình
* Hãy chắc chắn đã cài đủ 3 gói này:
* 1. moremata (Thư viện toán)
* 2. synth (Gói lõi của Abadie)
* 3. synth2 (Gói bọc (wrapper) hiện đại để chạy và vẽ hình)

* ssc install moremata, replace
* ssc install synth, replace
* ssc install synth2, replace
* ssc install allplot, replace // Hỗ trợ đồ thị cho synth2

*----------------------------------------------------------
* 2. MÔ PHỎNG DỮ LIỆU CẤP TỈNH (AGGREGATE PANEL DATA)
*----------------------------------------------------------
* SCM không dùng cho cá nhân (N=5000), mà dùng cho đơn vị (Unit).
* Giả định: 20 Tỉnh trong 20 Năm.
* Tỉnh số 1 (Province 1) áp dụng chính sách năm 2011.
* 19 Tỉnh còn lại (2-20) là Donor Pool.

set obs 20
gen province_id = _n
label define prov_lbl 1 "Tỉnh Can Thiệp (Treated)" 2 "Donor A" 3 "Donor B"
label values province_id prov_lbl

* Mở rộng theo thời gian (2001 - 2020)
expand 20
bysort province_id: gen year = 2000 + _n
xtset province_id year

* 2.1. Tạo các biến giải thích (Predictors)
* Để SCM hoạt động tốt, "Tỉnh nhân tạo" phải giống "Tỉnh thật" 
* về các đặc điểm kinh tế vĩ mô.
gen investment = rnormal(20, 5) + 0.5 * province_id
gen industry = runiform(10, 30)
gen educ_level = rnormal(10, 2)

* 2.2. Tạo biến kết quả (GDP Growth) - Pre-treatment
* Bí mật của mô phỏng SCM: Tỉnh 1 phải được cấu thành từ các tỉnh khác.
* Giả sử: Tỉnh 1 giống hệt 60% Tỉnh 2 + 40% Tỉnh 3 + Sai số nhỏ.
* Nếu không làm thế này, Convex Hull sẽ thất bại (không tìm được tỉnh giống).

* Tạo baseline cho outcome
gen gdp_growth = 5 + 0.3*investment + 0.1*industry + 0.5*educ_level + rnormal(0, 1)

* Điều chỉnh Tỉnh 1 (Treated) cho nó khớp với Tỉnh 2 và 3 trước năm 2011
bysort year: replace gdp_growth = 0.6*gdp_growth[2] + 0.4*gdp_growth[3] + rnormal(0, 0.2) ///
             if province_id == 1 & year < 2011

* 2.3. Tạo hiệu ứng can thiệp (Treatment Effect)
* Bắt đầu từ 2011, chính sách giúp Tỉnh 1 tăng trưởng vượt bậc (+4 điểm).
gen treated_period = (year >= 2011)
gen treatment_effect = 4 * treated_period * (province_id == 1)

replace gdp_growth = gdp_growth + treatment_effect

* Thêm chút biến động ngẫu nhiên cho Donor Pool để thực tế hơn
replace gdp_growth = gdp_growth + rnormal(0, 0.5) if province_id > 1

label var gdp_growth "Tăng trưởng GDP (%)"
label var year "Năm"

*----------------------------------------------------------
* 3. PHÂN TÍCH SYNTHETIC CONTROL VỚI 'SYNTH2'
*----------------------------------------------------------
* Bước này rất hay bị lỗi "Nested optimization failed" nếu dữ liệu quá nhiễu.
* synth2 tự động xử lý phần việc nặng nhọc.

display ">>> BẮT ĐẦU CHẠY SCM <<<"
display "Lưu ý: Quá trình này có thể mất vài giây để tối ưu hóa ma trận trọng số."

* CÚ PHÁP LỆNH SYNTH2:
* synth2 [Biến_KQ] [Các_biến_giải_thích], 
*        trunit(ID_Tỉnh_Can_Thiệp) trperiod(Năm_bắt_đầu)
*        nested (sử dụng tối ưu hóa lồng - chuẩn mực)
*        fig_treat (vẽ đồ thị Real vs Synthetic)
*        fig_plac (vẽ kiểm định Placebo - cực ngầu)

synth2 gdp_growth investment industry educ_level, /// 
       trunit(1) trperiod(2011) ///
       nested ///
       frame(gdp_plot)
       // frame() để lưu data vẽ hình riêng, không ảnh hưởng data gốc

* -> HƯỚNG DẪN GIẢNG:
* Đồ thị 1 hiện ra: 
* - Đường liền nét: Tỉnh Can Thiệp thực tế (Tỉnh 1).
* - Đường đứt nét: Tỉnh nhân tạo (Synthetic Unit).
* - Giai đoạn Pre (2001-2010): Hai đường này PHẢI quấn lấy nhau (khít nhau). 
*   Nếu lệch nhau -> SCM thất bại, Donors không đại diện được.
* - Giai đoạn Post (2011-2020): Đường liền nét vọt lên -> Hiệu quả can thiệp.

*----------------------------------------------------------
* 4. SUY DIỄN THỐNG KÊ (INFERENCE)
*----------------------------------------------------------
* SCM không có p-value kiểu T-test truyền thống (vì N=1).
* Ta dùng "Placebo Test": Giả sử 19 tỉnh còn lại lần lượt là tỉnh được can thiệp
* để xem "vận may" có thể tạo ra kết quả giống Tỉnh 1 hay không.

display ">>> ĐANG CHẠY PLACEBO TEST (HOÁN VỊ) <<<"
display "Lớp học vui lòng chờ, máy đang giả lập can thiệp trên 19 tỉnh còn lại..."

* Thêm tùy chọn:
* plac: Kích hoạt chế độ placebo
* graph_plac: Vẽ đồ thị các đường placebo (thường gọi là "Spaghetti Graph")
* mspe_unit: Tính tỷ số MSPE (Mean Squared Prediction Error)

* Chạy lại lệnh với tùy chọn Placebo (Lưu ý: sẽ hơi lâu)
synth2 gdp_growth investment industry educ_level, trunit(1) trperiod(2011) nested placebo(unit cut(2))

* -> HƯỚNG DẪN GIẢNG ĐỒ THỊ 2 (SPAGHETTI GRAPH):
* - Bạn sẽ thấy một "mớ bòng bong" các đường màu xám (Placebo gaps) dao động quanh số 0.
* - Đường của Tỉnh 1 (thường được tô đậm/khác màu) phải nằm VƯỢT TRỘI ra khỏi đám mây xám đó.
* - Ý nghĩa: "Xác suất để một tỉnh bất kỳ có được sự tăng trưởng này là cực thấp (p < 0.05)."

*----------------------------------------------------------
* 5. HIỂN THỊ TRỌNG SỐ (WEIGHTS TABLE)
*----------------------------------------------------------
* Học viên cần biết "Tỉnh nhân tạo" được xây dựng từ ai?
* Nó là hỗn hợp của những tỉnh nào?

* Sau lệnh synth2, kết quả lưu trong các return list
ereturn list


*=============================================================================
* KẾT THÚC BÀI GIẢNG SCM
*=============================================================================