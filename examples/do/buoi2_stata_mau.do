/* ============================================================
   DO FILE MẪU – BUỔI 2: PHÂN TÍCH KINH TẾ LƯỢNG VỚI STATA
   Khóa học: Stata cho Nghiên cứu Kinh tế (2 buổi)
   Giảng viên: TS. Đỗ Ngọc Kiên
   Bộ môn Kinh tế & Quản lý – ĐH Ngoại Thương

   MÔ TẢ: Do file ĐỘC LẬP – chỉ cần bấm Ctrl+D là chạy toàn bộ.
   Tự tạo dữ liệu mô phỏng 63 tỉnh × 10 năm (2013–2022).
   Bao gồm: OLS, Chẩn đoán, Biến giả & Tương tác,
             Panel FE/RE, Hausman, IV/2SLS, DiD, Xuất kết quả.

   CHỦ ĐỀ MÔ PHỎNG: Tác động FDI đến tăng trưởng kinh tế địa phương
   ============================================================ */


/* ============================================================
   PHẦN 0: KHỞI TẠO & TẠO DỮ LIỆU MÔ PHỎNG
   ============================================================ */

clear all
set more off
capture log close
set seed 2024        // Đảm bảo kết quả tái lặp được trên mọi máy

log using "buoi2_stata_mau.log", replace text
display "=== Buoi 2 – Phan tich Kinh te luong: $S_DATE $S_TIME ==="

* ---- Thiết lập xuất biểu đồ PNG ----
* Tất cả biểu đồ sẽ được lưu vào thư mục con "figures/"
capture mkdir "figures"
* Scheme đồ thị: s2color (mặc định), hoặc thử: s1color, lean2, white
set scheme s2color
* Kích thước chuẩn cho publication (width x height tính bằng pixel)
global png_opts "as(png) width(1600) height(1200) replace"
display "==> Bieu do se duoc luu vao thu muc: figures/"

* ---- Thông số dữ liệu ----
local N_prov = 63
local N_year = 10    // 2013–2022 (thêm 2 năm để demo DiD tốt hơn)
set obs `= `N_prov' * `N_year''

* ---- Biến định danh ----
gen province = ceil(_n / `N_year')
gen year     = 2012 + mod(_n - 1, `N_year') + 1

* ---- Vùng kinh tế ----
gen vung = .
replace vung = 1 if inrange(province, 50, 63)   // TD & MN phía Bắc
replace vung = 2 if inrange(province,  1, 11)   // Đồng bằng sông Hồng
replace vung = 3 if inrange(province, 12, 17)   // Bắc Trung Bộ
replace vung = 4 if inrange(province, 18, 25)   // Nam Trung Bộ
replace vung = 5 if inrange(province, 26, 30)   // Tây Nguyên
replace vung = 6 if inrange(province, 31, 49)   // ĐNB & ĐBSCL

label define vung_lbl 1 "TD&MN phia Bac" 2 "DBSH"          ///
                      3 "Bac Trung Bo"   4 "Nam Trung Bo"   ///
                      5 "Tay Nguyen"     6 "DNB & DBSCL"
label values vung vung_lbl
label variable vung "Vung kinh te"

* ---- Đô thị lớn ----
gen urban = (inlist(province, 1, 6, 18, 36, 45))
label variable urban "Thanh pho lon (1=Yes)"

* ============================================================
* TẠO BIẾN VỚI CẤU TRÚC TRUE DGP:
*   ln_gdp = a_i + b_t + 0.35*ln_fdi + 0.20*trade_share
*            + 0.15*ln_pop + 0.10*edu + u_it
* ============================================================

* Hiệu ứng tỉnh (province fixed effect – không quan sát được)
gen alpha_i = 2.0 + 0.8*(vung==2) + 1.2*(vung==6) ///
              + 1.5*urban + rnormal(0, 0.3)

* Hiệu ứng năm (year trend)
gen lambda_t = 0.04 * (year - 2012)

* ---- GDP bình quân đầu người (triệu VND, dạng mức) ----
gen gdp = exp(alpha_i + lambda_t + rnormal(0, 0.15)) * 30
replace gdp = round(gdp, 0.1)
gen ln_gdp = ln(gdp)
label variable gdp    "GDP binh quan dau nguoi (trieu VND)"
label variable ln_gdp "Log GDP binh quan dau nguoi"

* ---- FDI (triệu USD) – CÓ TƯƠNG QUAN VỚI alpha_i (nội sinh) ----
* FDI cao hơn ở tỉnh giàu → gây endogeneity cho OLS
gen fdi_raw = max(0.5, exp(0.6*alpha_i + 0.3*lambda_t ///
              + 20*(vung==2) + 30*(vung==6) + 50*urban ///
              + rnormal(0, 0.4)) )
replace fdi_raw = round(fdi_raw, 0.1)
gen ln_fdi = ln(fdi_raw + 1)
label variable fdi_raw "Von FDI (trieu USD)"
label variable ln_fdi  "Log(FDI + 1)"

* ---- Thương mại / Trade share (% GDP) ----
gen trade_share = 30 + 10*(vung==2) + 8*(vung==6) ///
                  + 5*urban + (year-2012)*1.5 + rnormal(0, 5)
replace trade_share = min(max(trade_share, 5), 95)
replace trade_share = round(trade_share, 0.1)
label variable trade_share "Kim ngach XNK / GDP (%)"

* ---- Dân số (nghìn người) ----
gen pop = 400 + province*15 + 3000*urban + rnormal(0, 80)
replace pop = max(pop, 100)
replace pop = round(pop, 1)
gen ln_pop = ln(pop)
label variable pop    "Dan so (nghin nguoi)"
label variable ln_pop "Log dan so"

* ---- Tỷ lệ lao động qua đào tạo / Education (%) ----
gen edu = 12 + 8*(vung==2) + 6*(vung==4) + 14*urban ///
          + (year-2012)*0.8 + rnormal(0, 2)
replace edu = min(max(edu, 3), 80)
replace edu = round(edu, 0.1)
label variable edu "Ti le lao dong qua dao tao (%)"

* ---- Cơ sở hạ tầng (chỉ số tổng hợp 0–100) ----
gen infra = 20 + 10*(vung==2) + 8*(vung==6) + 20*urban ///
            + (year-2012)*1.2 + rnormal(0, 4)
replace infra = min(max(infra, 5), 95)
replace infra = round(infra, 0.1)
label variable infra "Chi so co so ha tang (0-100)"

* ---- BIẾN CÔNG CỤ cho FDI ----
* Z1: Khoảng cách đến cảng biển gần nhất (km) – time-invariant, exogenous
*     Tác động đến FDI, không tác động trực tiếp đến GDP
gen dist_port = 50 + province*3 - 15*(inlist(province,6,36,18,45)) ///
               + rnormal(0, 10)
replace dist_port = max(dist_port, 10)
replace dist_port = round(dist_port, 1)
gen ln_dist = ln(dist_port)
label variable dist_port "Khoang cach den cang bien (km)"
label variable ln_dist   "Log khoang cach den cang bien"

* Z2: Chi phí điện bình quân (USD/kWh × 100) – phần ngoại sinh
gen elec_cost = 8 - 0.02*infra + rnormal(0, 0.5)
replace elec_cost = max(elec_cost, 3)
replace elec_cost = round(elec_cost, 0.1)
label variable elec_cost "Chi phi dien (USD/kWh x 100)"

* ---- CHÍNH SÁCH DiD: Khu kinh tế tự do ----
* Năm 2018: 20 tỉnh ở vùng 6 và vùng 2 được chọn nhận chính sách
gen treated = (inrange(province, 1, 8) | inrange(province, 31, 42))
gen post    = (year >= 2018)
gen did_var = treated * post
label variable treated "Tinh nhan chinh sach (DiD)"
label variable post    "Sau chinh sach (year>=2018)"
label variable did_var "Tuong tac DiD (treated x post)"

* Thêm hiệu ứng xử lý thực sự (ATT = 0.08) vào ln_gdp
replace ln_gdp = ln_gdp + 0.08 * did_var + rnormal(0, 0.02) * did_var
replace gdp    = exp(ln_gdp)

* ---- Thêm một số missing để thực tế hơn ----
replace ln_fdi = . if runiform() < 0.015
replace edu    = . if runiform() < 0.01

* ---- Nhãn & sắp xếp ----
label data "Panel 63 tinh VN 2013-2022 – Mo phong Buoi 2 (seed=2024)"
notes: Du lieu hoan toan mo phong. True ATT (DiD) = 0.08. True beta_FDI = 0.35
sort province year

display "=== Da tao du lieu: `= _N' quan sat (`N_prov' tinh x `N_year' nam) ==="

* Lưu để dùng lại trong các phần sau
save "panel_buoi2.dta", replace


/* ============================================================
   PHẦN 0B: BIỂU ĐỒ KHÁM PHÁ DỮ LIỆU (EDA Charts)
   ============================================================ */

display _newline(2) "===== PHẦN 0B: BIEU DO KHAM PHA DU LIEU ====="

* --- Biểu đồ 1: Phân phối GDP bình quân ---
histogram ln_gdp, normal kdensity ///
    title("Phan phoi Log GDP binh quan dau nguoi") ///
    xtitle("ln(GDP binh quan)") ytitle("Mat do") ///
    note("N = 630 quan sat (63 tinh x 10 nam, 2013-2022)")
graph export "figures/01_histogram_lngdp.png", $png_opts
display "✓ figures/01_histogram_lngdp.png"

* --- Biểu đồ 2: So sánh GDP theo vùng kinh tế (box plot) ---
graph box ln_gdp, over(vung, label(angle(30))) ///
    title("Phan phoi GDP binh quan theo Vung kinh te") ///
    ytitle("ln(GDP binh quan dau nguoi)") ///
    note("Duong ngang trong hop = Trung vi | Dau cham = Outlier")
graph export "figures/02_boxplot_gdp_vung.png", $png_opts
display "✓ figures/02_boxplot_gdp_vung.png"

* --- Biểu đồ 3: Xu hướng GDP theo năm (theo vùng) ---
preserve
collapse (mean) ln_gdp, by(year vung)
twoway (line ln_gdp year if vung==1, lcolor(navy)    lwidth(medthick)) ///
       (line ln_gdp year if vung==2, lcolor(maroon)  lwidth(medthick)) ///
       (line ln_gdp year if vung==3, lcolor(forest_green) lwidth(medthick)) ///
       (line ln_gdp year if vung==4, lcolor(dkorange) lwidth(medthick)) ///
       (line ln_gdp year if vung==5, lcolor(purple)   lwidth(medthick)) ///
       (line ln_gdp year if vung==6, lcolor(teal)     lwidth(medthick)), ///
    legend(label(1 "TD&MN phia Bac") label(2 "DBSH") ///
           label(3 "Bac Trung Bo")   label(4 "Nam Trung Bo") ///
           label(5 "Tay Nguyen")     label(6 "DNB & DBSCL") ///
           rows(2) size(small)) ///
    title("Xu huong GDP binh quan theo Vung kinh te (2013-2022)") ///
    xtitle("Nam") ytitle("ln(GDP binh quan dau nguoi)") ///
    xlabel(2013(1)2022, angle(45))
graph export "figures/03_trend_gdp_by_vung.png", $png_opts
restore
display "✓ figures/03_trend_gdp_by_vung.png"

* --- Biểu đồ 4: Scatter FDI vs GDP (với màu theo vùng) ---
twoway (scatter ln_gdp ln_fdi if vung==1, mcolor(navy)    msymbol(o)  msize(small)) ///
       (scatter ln_gdp ln_fdi if vung==2, mcolor(maroon)  msymbol(d)  msize(small)) ///
       (scatter ln_gdp ln_fdi if vung==3, mcolor(forest_green) msymbol(t) msize(small)) ///
       (scatter ln_gdp ln_fdi if vung==4, mcolor(dkorange) msymbol(s)  msize(small)) ///
       (scatter ln_gdp ln_fdi if vung==5, mcolor(purple)   msymbol(oh) msize(small)) ///
       (scatter ln_gdp ln_fdi if vung==6, mcolor(teal)     msymbol(dh) msize(small)) ///
       (lfit    ln_gdp ln_fdi,            lcolor(red) lwidth(medthick)), ///
    legend(label(1 "TD&MN phia Bac") label(2 "DBSH") ///
           label(3 "Bac Trung Bo")   label(4 "Nam Trung Bo") ///
           label(5 "Tay Nguyen")     label(6 "DNB & DBSCL") ///
           label(7 "Duong xu the OLS") rows(3) size(small)) ///
    title("Moi quan he FDI va GDP binh quan") ///
    xtitle("ln(FDI + 1)") ytitle("ln(GDP binh quan dau nguoi)") ///
    note("Moi diem = 1 tinh x 1 nam. Duong do = OLS fit.")
graph export "figures/04_scatter_fdi_gdp.png", $png_opts
display "✓ figures/04_scatter_fdi_gdp.png"


/* ============================================================
   PHẦN 1: HỒI QUY OLS CƠ BẢN
   ============================================================ */

display _newline(2) "===== PHẦN 1: HỒI QUY OLS CƠ BẢN ====="

use "panel_buoi2.dta", clear

* --- 1.1 OLS đơn giản (không kiểm soát gì) ---
display _newline "--- 1.1 OLS don gian ---"
regress ln_gdp ln_fdi
/*
   Kết quả: hệ số ln_fdi bị chệch lên vì FDI tương quan với alpha_i
   (tỉnh giàu có cả GDP cao lẫn FDI cao → omitted variable bias)
*/

* --- 1.2 OLS đầy đủ biến kiểm soát ---
display _newline "--- 1.2 OLS day du bien kiem soat ---"
regress ln_gdp ln_fdi trade_share ln_pop edu infra
ereturn list        // Xem toàn bộ scalar lưu sau hồi quy
display "R-squared = " %6.4f e(r2)
display "N         = " e(N)

* --- 1.3 Thêm Year Fixed Effects bằng biến giả i.year ---
display _newline "--- 1.3 OLS + Year FE (i.year) ---"
regress ln_gdp ln_fdi trade_share ln_pop edu i.year
/*
   i.year: Stata tự tạo biến giả cho từng năm, bỏ năm đầu làm base
   Kiểm soát shocks chung theo thời gian (ví dụ: khủng hoảng, COVID)
*/

* --- 1.4 Lưu giá trị dự báo và phần dư ---
display _newline "--- 1.4 Luu gia tri du bao va phan du ---"
regress ln_gdp ln_fdi trade_share ln_pop edu i.year
predict yhat,   xb          // Giá trị dự báo (fitted values)
predict resid,  residuals   // Phần dư

summarize yhat resid
list province year ln_gdp yhat resid in 1/8

* --- 1.5 Hệ số hồi quy chuẩn hóa (beta coefficients) ---
display _newline "--- 1.5 Beta (standardized) coefficients ---"
regress ln_gdp ln_fdi trade_share ln_pop edu, beta
/*
   Beta cho phép so sánh độ lớn tác động giữa các biến
   có đơn vị đo khác nhau
*/

drop yhat resid


/* ============================================================
   PHẦN 2: KIỂM ĐỊNH CHẨN ĐOÁN & ROBUST STANDARD ERRORS
   ============================================================ */

display _newline(2) "===== PHẦN 2: CHẨN ĐOÁN & ROBUST SE ====="

* Chạy OLS gốc để làm nền cho kiểm định
regress ln_gdp ln_fdi trade_share ln_pop edu i.year

* --- 2.1 Breusch-Pagan: phương sai không đều ---
display _newline "--- 2.1 Breusch-Pagan (Heteroskedasticity) ---"
estat hettest
/*
   H0: Phương sai đồng đều (homoskedasticity)
   p < 0.05 → bác bỏ H0 → CÓ phương sai không đều → dùng robust SE
*/

* --- 2.2 Ramsey RESET: kiểm định dạng hàm ---
display _newline "--- 2.2 Ramsey RESET (dang ham sai) ---"
estat ovtest
/*
   H0: Dạng hàm đúng
   p < 0.05 → có thể bỏ sót biến phi tuyến
*/

* --- 2.3 VIF: Đa cộng tuyến ---
display _newline "--- 2.3 VIF (Da cong tuyen) ---"
regress ln_gdp ln_fdi trade_share ln_pop edu   // Bỏ i.year để VIF rõ hơn
vif
/*
   VIF > 10: vấn đề nghiêm trọng
   VIF 5–10: cần chú ý
   VIF < 5:  bình thường
*/

* --- 2.4 Kiểm tra phần dư ---
display _newline "--- 2.4 Kiem tra phan du ---"
regress ln_gdp ln_fdi trade_share ln_pop edu i.year
predict resid2, residuals

histogram resid2, normal title("Phan du – OLS co ban") ///
    xtitle("Phan du") ytitle("Mat do") ///
    note("Nen xap xi phan phoi chuan neu mo hinh tot")
graph export "figures/05_histogram_resid_ols.png", $png_opts
display "✓ figures/05_histogram_resid_ols.png"

pnorm resid2          // P-P plot: phần dư có phân phối chuẩn?
graph export "figures/06_pnorm_resid_ols.png", $png_opts
display "✓ figures/06_pnorm_resid_ols.png"

rvfplot, yline(0) title("Residual vs Fitted (OLS)") ///
    xtitle("Gia tri du bao (yhat)") ytitle("Phan du") ///
    note("Phan du nen phan bo deu quanh duong 0 – khong co quy luat")
graph export "figures/07_rvfplot_ols.png", $png_opts
display "✓ figures/07_rvfplot_ols.png"

drop resid2

* --- 2.5 Heteroskedasticity-Robust Standard Errors ---
display _newline "--- 2.5 Robust SE (Huber-White) ---"
regress ln_gdp ln_fdi trade_share ln_pop edu i.year, robust
/*
   SE thay đổi nhưng hệ số KHÔNG thay đổi
   Robust SE nhất quán khi có phương sai không đều
*/

* --- 2.6 Cluster-Robust SE (khuyến nghị cho panel) ---
display _newline "--- 2.6 Cluster-Robust SE (theo tinh) ---"
regress ln_gdp ln_fdi trade_share ln_pop edu i.year, ///
    vce(cluster province)
/*
   Cluster theo province: kiểm soát tương quan chuỗi thời gian
   trong cùng tỉnh. Khuyến nghị mạnh cho dữ liệu bảng.
*/


/* ============================================================
   PHẦN 3: FACTOR VARIABLES, BIẾN GIẢ & TƯƠNG TÁC
   ============================================================ */

display _newline(2) "===== PHẦN 3: BIẾN GIẢ & TƯƠNG TÁC ====="

* --- 3.1 Biến giả phân loại tự động (i.var) ---
display _newline "--- 3.1 Bien gia vung kinh te (i.vung) ---"
regress ln_gdp ln_fdi trade_share edu i.vung, robust
/*
   Stata tự tạo 5 biến giả (bỏ vung==1 làm base category)
   Hệ số của _Ivung_k = chênh lệch so với vùng 1
*/

* Đổi nhóm base category: vùng 2 (ĐBSH) làm cơ sở
regress ln_gdp ln_fdi trade_share edu ib2.vung, robust
/*
   ib2.vung: "i" = factor, "b2" = base category là nhóm 2
*/

* --- 3.2 Biến giả năm + vùng đồng thời ---
display _newline "--- 3.2 Bien gia nam + vung ---"
regress ln_gdp ln_fdi trade_share edu i.year i.vung, robust

* --- 3.3 Tương tác phân loại × liên tục (i.vung#c.ln_fdi) ---
display _newline "--- 3.3 Tuong tac vung x FDI ---"
regress ln_gdp c.ln_fdi#i.vung trade_share edu i.year, robust
/*
   Hệ số riêng của ln_fdi cho từng vùng kinh tế
   Câu hỏi: Tác động FDI có khác nhau giữa các vùng không?
*/

* --- 3.4 Tương tác đầy đủ (i.vung##c.ln_fdi) ---
display _newline "--- 3.4 Tuong tac day du (##) ---"
regress ln_gdp i.vung##c.ln_fdi trade_share edu i.year, robust
/*
   ## = hiệu ứng chính (main effect) + tương tác (interaction)
   Tương đương với: i.vung c.ln_fdi c.ln_fdi#i.vung
*/

* --- 3.5 margins: Hiệu ứng biên sau tương tác ---
display _newline "--- 3.5 margins – hieu ung bien cua FDI theo vung ---"
regress ln_gdp i.vung##c.ln_fdi trade_share edu i.year, robust
margins, dydx(ln_fdi)              // Trung bình toàn mẫu
margins vung, dydx(ln_fdi)         // Theo từng vùng
marginsplot, title("Hieu ung bien cua FDI theo Vung kinh te") ///
    ytitle("dln_gdp / dln_fdi") xtitle("Vung kinh te") ///
    note("AME: Average Marginal Effect. Thanh = Khoang tin cay 95%")
graph export "figures/08_margins_fdi_by_vung.png", $png_opts
display "✓ figures/08_margins_fdi_by_vung.png"
/*
   margins đưa ra hiệu ứng biên trung bình (AME) đúng hơn
   việc đọc hệ số trực tiếp khi có tương tác
*/

* --- 3.6 Tương tác liên tục × liên tục ---
display _newline "--- 3.6 Tuong tac FDI x Co so ha tang ---"
regress ln_gdp c.ln_fdi##c.infra trade_share edu i.year, robust
/*
   Câu hỏi: FDI có hiệu quả hơn ở những nơi hạ tầng tốt không?
   Hệ số c.ln_fdi#c.infra: hiệu ứng bổ sung
*/
margins, dydx(ln_fdi) at(infra=(20 40 60 80))
marginsplot, title("Hieu ung FDI tai cac muc co so ha tang khac nhau") ///
    xtitle("Muc co so ha tang (infra)") ytitle("Hieu ung bien cua FDI") ///
    note("FDI hieu qua hon o nhung noi co ha tang tot hon")
graph export "figures/09_margins_fdi_by_infra.png", $png_opts
display "✓ figures/09_margins_fdi_by_infra.png"


/* ============================================================
   PHẦN 4: DỮ LIỆU BẢNG – PANEL DATA (xtset, xtsum, xtreg)
   ============================================================ */

display _newline(2) "===== PHẦN 4: PANEL DATA ====="

use "panel_buoi2.dta", clear

* --- 4.1 Khai báo cấu trúc panel ---
display _newline "--- 4.1 xtset – Khai bao panel ---"
xtset province year
/*
   province: panelvar (đơn vị chéo)
   year:     timevar  (chiều thời gian)
   Stata sẽ tự nhận dạng L. F. D. operators
*/

* --- 4.2 Kiểm tra cân bằng ---
display _newline "--- 4.2 xtdescribe – Kiem tra can bang ---"
xtdescribe
/*
   Balanced: mỗi tỉnh có đúng T quan sát
   Unbalanced: số quan sát không đều nhau
*/

* --- 4.3 Phân tách biến thiên: Overall / Between / Within ---
display _newline "--- 4.3 xtsum – Phan tach bien thien ---"
xtsum ln_gdp ln_fdi trade_share edu
/*
   Overall SD  = biến thiên toàn mẫu
   Between SD  = biến thiên GIỮA các tỉnh (cross-section variation)
   Within SD   = biến thiên TRONG tỉnh theo thời gian
   
   SD_within nhỏ → biến ít thay đổi theo thời gian (khó FE ước lượng)
   SD_between nhỏ → biến ít khác nhau giữa các tỉnh (khó RE ước lượng)
*/

* --- 4.4 Biến trễ (lag), lead, sai phân ---
display _newline "--- 4.4 Bien tre, sai phan ---"
gen L1_gdp  = L.ln_gdp     // Lag 1 kỳ
gen L2_gdp  = L2.ln_gdp    // Lag 2 kỳ
gen D_gdp   = D.ln_gdp     // Sai phân bậc 1 (Δln_gdp)
gen F1_gdp  = F.ln_gdp     // Lead 1 kỳ (tương lai)

summarize ln_gdp L1_gdp L2_gdp D_gdp
list province year ln_gdp L1_gdp D_gdp if province == 1

drop L1_gdp L2_gdp D_gdp F1_gdp


/* ============================================================
   PHẦN 5: FIXED EFFECTS & RANDOM EFFECTS
   ============================================================ */

display _newline(2) "===== PHẦN 5: FE & RE ====="

* --- 5.1 Pooled OLS (bỏ qua cấu trúc panel – sai) ---
display _newline "--- 5.1 Pooled OLS (khong kiem soat FE) ---"
regress ln_gdp ln_fdi trade_share edu, robust
/*
   Hệ số bị chệch vì omitted variable bias (alpha_i)
   ln_fdi ước lượng được ~ 0.5–0.6 thay vì giá trị thật 0.35
*/

* --- 5.2 Fixed Effects (FE / within estimator) ---
display _newline "--- 5.2 Fixed Effects (xtreg, fe) ---"
xtreg ln_gdp ln_fdi trade_share edu, fe
/*
   fe: loại bỏ alpha_i bằng within-transformation (demeaning)
   Hệ số ln_fdi gần với giá trị thật hơn Pooled OLS
   F-test cuối: kiểm định tất cả u_i = 0
*/

xtreg ln_gdp ln_fdi trade_share edu, fe robust
/*
   robust: heteroskedasticity-robust SE
*/

* Two-way FE: kiểm soát cả province FE và year FE
display _newline "--- Two-way Fixed Effects ---"
xtreg ln_gdp ln_fdi trade_share edu i.year, fe vce(cluster province)
/*
   i.year bên trong xtreg fe: thêm year dummies
   vce(cluster province): cluster-robust SE – khuyến nghị nhất
*/
estimates store fe_twoway
display "==> Ket qua Two-way FE da luu vao 'fe_twoway'"

* --- 5.3 Random Effects ---
display _newline "--- 5.3 Random Effects (xtreg, re) ---"
xtreg ln_gdp ln_fdi trade_share edu i.year, re
/*
   re: giả sử alpha_i ~ iid, không tương quan với X_it
   Ước lượng GLS, tận dụng cả between và within variation
   Có thể đưa biến time-invariant (vd: vung, urban) vào RE
*/
estimates store re_model

* RE có biến time-invariant
xtreg ln_gdp ln_fdi trade_share edu urban i.vung i.year, re
display "==> RE co them bien time-invariant (urban, vung)"

* --- 5.4 Between Effects (chỉ dùng variation giữa các tỉnh) ---
display _newline "--- 5.4 Between Effects (xtreg, be) ---"
xtreg ln_gdp ln_fdi trade_share edu, be
/*
   be: hồi quy trung bình tỉnh – chỉ khai thác cross-section
   Ít dùng trong nghiên cứu hiện đại
*/


/* ============================================================
   PHẦN 6: HAUSMAN TEST & LỰA CHỌN MÔ HÌNH
   ============================================================ */

display _newline(2) "===== PHẦN 6: HAUSMAN TEST ====="

* --- 6.1 Quy trình chuẩn: FE → RE → Hausman ---
display _newline "--- 6.1 Quy trinh Hausman ---"

* Bước 1: Chạy FE và lưu
xtreg ln_gdp ln_fdi trade_share edu, fe
estimates store fe_model

* Bước 2: Chạy RE và lưu
xtreg ln_gdp ln_fdi trade_share edu, re
estimates store re_model

* Bước 3: Hausman test
hausman fe_model re_model
/*
   H0: RE nhất quán và hiệu quả (alpha_i không tương quan với X)
   p < 0.05 → bác bỏ H0 → CHỌN Fixed Effects
   p ≥ 0.05 → không bác bỏ → Random Effects có thể dùng

   Với dữ liệu mô phỏng này: FDI tương quan với alpha_i
   → Kỳ vọng p < 0.05 → FE được ưu tiên
*/

* --- 6.2 Correlated Random Effects (Mundlak, 1978) ---
display _newline "--- 6.2 Mundlak RE (Correlated RE) ---"
* Thêm trung bình theo tỉnh của các biến giải thích vào RE
bysort province: egen mean_fdi  = mean(ln_fdi)
bysort province: egen mean_edu  = mean(edu)
bysort province: egen mean_trd  = mean(trade_share)

xtreg ln_gdp ln_fdi trade_share edu mean_fdi mean_edu mean_trd, re
/*
   Nếu hệ số của mean_X có ý nghĩa thống kê → FE phù hợp hơn
   Mundlak giúp đưa biến time-invariant vào mô hình "FE-like"
*/

* Kiểm định ý nghĩa đồng thời của các mean terms
test mean_fdi mean_edu mean_trd
/*
   p < 0.05 → các mean term có ý nghĩa → FE tốt hơn RE
*/

drop mean_fdi mean_edu mean_trd

* --- 6.3 So sánh Pooled OLS vs FE vs RE ---
display _newline "--- 6.3 Bang so sanh: Pooled vs FE vs RE ---"
eststo clear
eststo m_ols: regress  ln_gdp ln_fdi trade_share edu
eststo m_fe:  xtreg    ln_gdp ln_fdi trade_share edu, fe robust
eststo m_re:  xtreg    ln_gdp ln_fdi trade_share edu, re

esttab m_ols m_fe m_re, ///
    title("So sanh: Pooled OLS vs FE vs RE") ///
    mtitles("Pooled OLS" "Fixed Effects" "Random Effects") ///
    star(* 0.1 ** 0.05 *** 0.01) b(3) se(3) r2 scalars(N) ///
    note("True beta_FDI = 0.35 (theo DGP mo phong)")
/*
   Quan sát: hệ số ln_fdi giảm dần từ Pooled → RE → FE
   vì FE loại bỏ omitted variable bias tốt nhất
*/


/* ============================================================
   PHẦN 7: BIẾN CÔNG CỤ – IV / 2SLS  (dùng ivreg2)
   Yêu cầu: ssc install ivreg2
   ============================================================ */

display _newline(2) "===== PHẦN 7: IV / 2SLS (ivreg2) ====="

* Vấn đề: FDI nội sinh vì tỉnh giàu thu hút FDI nhiều hơn
* Biến công cụ: ln_dist (khoảng cách đến cảng) + elec_cost
* Lý thuyết:
*   Relevance: Tỉnh gần cảng, điện rẻ → thu hút FDI nhiều hơn ✓
*   Exclusion: Khoảng cách cảng, giá điện không tác động trực tiếp đến GDP ✓

* Cú pháp ivreg2:
*   ivreg2 Y X_exog (X_endo = Z1 Z2), [options]
*   Biến nội sinh và IV đặt trong ngoặc đơn CÙNG VỚI DẤU =
*   Khác ivregress: biến nội sinh nằm bên trái dấu =, IV bên phải

* --- 7.1 OLS đơn giản (để so sánh) ---
display _newline "--- 7.1 OLS (de so sanh voi IV) ---"
regress ln_gdp ln_fdi trade_share edu, robust
estimates store ols_naive
display "OLS beta_FDI (bi chech): " _b[ln_fdi]

* --- 7.2 IV/2SLS cơ bản với ivreg2 ---
display _newline "--- 7.2 IV/2SLS (ivreg2) ---"
ivreg2 ln_gdp trade_share edu (ln_fdi = ln_dist elec_cost), ///
    robust
estimates store iv_2sls
/*
   Cú pháp: ivreg2 Y X_exog (X_endo = Z1 Z2), [options]
   - trade_share edu: biến ngoại sinh (viết tự do ngoài ngoặc)
   - (ln_fdi = ln_dist elec_cost): ln_fdi nội sinh, IV là ln_dist elec_cost
   - robust: Heteroskedasticity-robust SE (Huber-White)

   ivreg2 tự động báo cáo sau khi chạy:
     · Kleibergen-Paap rk Wald F-stat  → kiểm định weak instruments
     · Hansen J-stat (nếu over-id)     → kiểm định exclusion restriction
*/
display "IV beta_FDI (nhat quan hon): " _b[ln_fdi]

* --- 7.3 Cluster-robust SE (khuyến nghị cho panel) ---
display _newline "--- 7.3 ivreg2 voi Cluster-robust SE ---"
ivreg2 ln_gdp trade_share edu (ln_fdi = ln_dist elec_cost), ///
    cluster(province)
/*
   cluster(province): SE cụm theo tỉnh, kiểm soát tương quan
   chuỗi thời gian trong cùng tỉnh – khuyến nghị mạnh cho panel.
   Khi dùng cluster, ivreg2 báo cáo:
     · Kleibergen-Paap rk Wald F-stat (robust hơn Cragg-Donald
       khi có cluster SE, không như ivregress/estat firststage)
     · Hansen J-stat thay cho Sargan (robust với cluster)
*/
estimates store iv_cluster

* --- 7.4 Xem kết quả kiểm định tích hợp ---
display _newline "--- 7.4 Ket qua kiem dinh tich hop trong ivreg2 ---"
/*
   ivreg2 in sẵn các kiểm định quan trọng cuối bảng kết quả:

   [A] Underidentification test (Kleibergen-Paap rk LM statistic)
       H0: Mô hình under-identified (IV không tương quan với X_endo)
       p < 0.05 → IV có tương quan → mô hình được xác định ✓

   [B] Weak identification test (Kleibergen-Paap rk Wald F-stat)
       Rule of thumb: F > 10 → IV đủ mạnh
       So sánh với ngưỡng Stock-Yogo (in kèm trong output)

   [C] Hansen J-stat (Over-identification, chỉ khi #IV > #X_endo)
       H0: Tất cả IV ngoại sinh (exclusion restriction đúng)
       p > 0.05 → không bác bỏ H0 → IV hợp lệ ✓
       p < 0.05 → nghi ngờ ít nhất một IV không hợp lệ

   Lưu ý: Không cần gọi estat firststage / estat overid / estat endogenous
   riêng như với ivregress – ivreg2 tích hợp tất cả.
*/

* --- 7.5 Kiểm định nội sinh (endogeneity) với endog() ---
display _newline "--- 7.5 Kiem dinh noi sinh (C-test / endog) ---"
ivreg2 ln_gdp trade_share edu (ln_fdi = ln_dist elec_cost), ///
    cluster(province) endog(ln_fdi)
/*
   endog(ln_fdi): C-test kiểm định ln_fdi có thực sự nội sinh không
   H0: ln_fdi ngoại sinh (OLS nhất quán)
   p < 0.05 → bác bỏ H0 → FDI nội sinh → nên dùng IV
   p ≥ 0.05 → OLS không bị chệch → OLS có thể chấp nhận được

   Lưu ý phân biệt hai option trong ivreg2:
   · endog(X): kiểm định biến nội sinh X có thực sự cần IV không
               → đây là lựa chọn đúng để kiểm định endogeneity
   · orthog(Z): kiểm định một biến IV Z có thực sự ngoại sinh không
                (C-test / difference-in-Sargan), chỉ dùng khi over-identified
                VD: orthog(elec_cost) để xem elec_cost có hợp lệ không

   endog() mạnh hơn DWH (của ivregress/estat endogenous) vì
   robust với cluster SE và heteroskedasticity.
*/

* --- 7.5b Kiểm định tính hợp lệ của từng IV riêng lẻ (orthog) ---
display _newline "--- 7.5b C-test tung IV (orthog) ---"
ivreg2 ln_gdp trade_share edu (ln_fdi = ln_dist elec_cost), ///
    cluster(province) orthog(elec_cost)
/*
   orthog(elec_cost): C-test kiểm định elec_cost có ngoại sinh không
   (Chỉ chạy được khi over-identified: có 2 IV nhưng chỉ 1 biến nội sinh)
   H0: elec_cost ngoại sinh
   p > 0.05 → elec_cost hợp lệ làm IV ✓
   p < 0.05 → elec_cost nghi ngờ không hợp lệ → xem lại lý thuyết
*/

* --- 7.6 Hiển thị Stage 1 tường minh (tùy chọn) ---
display _newline "--- 7.6 Stage 1 tuong minh (first) ---"
ivreg2 ln_gdp trade_share edu (ln_fdi = ln_dist elec_cost), ///
    cluster(province) first
/*
   first: in đầy đủ kết quả hồi quy Giai đoạn 1
          ln_fdi = f(ln_dist, elec_cost, trade_share, edu)
   Kiểm tra: hệ số ln_dist và elec_cost phải có ý nghĩa thống kê
   và mang đúng dấu kỳ vọng (âm: tỉnh xa cảng / điện đắt → ít FDI)
*/

* --- 7.7 Panel IV với ivreg2 + absorb (reghdfe-style) ---
display _newline "--- 7.7 Panel IV: ivreg2 + FE theo tinh va nam ---"
xtset province year

* Cách 1: Đưa biến giả tỉnh và năm vào trực tiếp (phù hợp N nhỏ)
ivreg2 ln_gdp trade_share edu i.year i.province ///
    (ln_fdi = ln_dist elec_cost), ///
    cluster(province) small
/*
   i.province: hấp thụ province FE (tương đương xtreg, fe)
   i.year:     hấp thụ year FE
   small:      dùng phân phối F thay vì chi-squared (mẫu nhỏ/trung bình)
   → Kết hợp hai-way FE + IV + cluster SE trong một lệnh
   → Kleibergen-Paap F-stat vẫn được báo cáo đầy đủ
*/
estimates store fe_iv

* Cách 2: Dùng xtivreg, fe khi cần Panel-IV chuẩn (giữ lại để tham khảo)
* xtivreg ln_gdp (ln_fdi = ln_dist elec_cost) trade_share edu i.year, fe
* (xtivreg không báo cáo Kleibergen-Paap → kém hơn ivreg2 + i.province)

* --- 7.8 So sánh OLS, IV, FE, FE+IV ---
display _newline "--- 7.8 Bang so sanh cac phuong phap ---"
eststo clear
eststo m1: quietly regress ln_gdp ln_fdi trade_share edu, ///
               vce(cluster province)
eststo m2: quietly ivreg2 ln_gdp trade_share edu ///
               (ln_fdi = ln_dist elec_cost), cluster(province)
eststo m3: quietly xtreg ln_gdp ln_fdi trade_share edu, ///
               fe vce(cluster province)
eststo m4: quietly ivreg2 ln_gdp trade_share edu i.year i.province ///
               (ln_fdi = ln_dist elec_cost), cluster(province) small

esttab m1 m2 m3 m4, ///
    title("So sanh: OLS / IV (ivreg2) / FE / FE+IV (ivreg2)") ///
    mtitles("OLS" "2SLS" "FE" "FE+IV") ///
    keep(ln_fdi trade_share edu) ///
    star(* 0.1 ** 0.05 *** 0.01) b(3) se(3) ///
    note("True beta_FDI = 0.35 | IV: ln_dist elec_cost | SE chum theo tinh")
/*
   Bài học: OLS > IV > FE: hệ số giảm dần khi kiểm soát bias tốt hơn
   FE+IV thường gần nhất với giá trị thật khi có đủ IV mạnh
*/


/* ============================================================
   PHẦN 8: DIFFERENCE-IN-DIFFERENCES (DiD)
   ============================================================ */

display _newline(2) "===== PHẦN 8: DIFFERENCE-IN-DIFFERENCES ====="

use "panel_buoi2.dta", clear
xtset province year

* Nhắc lại thiết kế:
*   treated = 1: các tỉnh nhận chính sách Khu kinh tế tự do
*   post    = 1: sau năm 2018
*   True ATT = 0.08 (tức 8% tăng GDP)

* --- 8.1 DiD thủ công (bảng 2×2) ---
display _newline "--- 8.1 DiD 2x2 thu cong ---"
tabulate treated post, summarize(ln_gdp)
/*
   Tính ATT thủ công:
   ATT = (Y_treated,post - Y_treated,pre) - (Y_control,post - Y_control,pre)
*/

* Tính từng ô
quietly summarize ln_gdp if treated==1 & post==1
local T1 = r(mean)
quietly summarize ln_gdp if treated==1 & post==0
local T0 = r(mean)
quietly summarize ln_gdp if treated==0 & post==1
local C1 = r(mean)
quietly summarize ln_gdp if treated==0 & post==0
local C0 = r(mean)

display "DiD thu cong:"
display "  DID (treated): Y_post - Y_pre = " %6.4f (`T1' - `T0')
display "  DID (control): Y_post - Y_pre = " %6.4f (`C1' - `C0')
display "  ATT uoc tinh  = " %6.4f ((`T1'-`T0') - (`C1'-`C0'))
display "  True ATT      = 0.0800"

* --- 8.2 DiD hồi quy cơ bản ---
display _newline "--- 8.2 DiD hoi quy co ban ---"
regress ln_gdp treated post did_var, robust
/*
   treated: chênh lệch cố định giữa treated và control
   post:    xu hướng chung sau chính sách
   did_var: HIỆU ỨNG NHÂN QUẢ của chính sách (ATT)
*/

* --- 8.3 DiD với biến kiểm soát ---
display _newline "--- 8.3 DiD voi bien kiem soat ---"
regress ln_gdp did_var treated post trade_share edu infra, ///
    vce(cluster province)

* --- 8.4 DiD dạng bảng: Two-way FE (KHUYẾN NGHỊ) ---
display _newline "--- 8.4 DiD Two-way FE (khong can treated, post) ---"
xtreg ln_gdp did_var trade_share edu infra i.year, ///
    fe vce(cluster province)
estimates store did_fe
/*
   Province FE hấp thụ biến 'treated' (time-invariant)
   Year FE hấp thụ biến 'post' (common trend)
   Hệ số did_var = ATT thuần túy (nên gần 0.08)
*/
display "ATT uoc tinh (FE): " %6.4f _b[did_var]
display "True ATT:           0.0800"

* --- 8.5 Event Study (kiểm định Parallel Trends) ---
display _newline "--- 8.5 Event Study (Parallel Trends Check) ---"

* Năm can thiệp = 2018 → relative year = year - 2018
gen rel_year = year - 2018

* Tạo biến sự kiện cho từng khoảng thời gian (-4 đến +4)
* Bỏ rel_year == -1 làm base (năm ngay trước chính sách)
forvalues k = 4(-1)2 {
    gen pre_`k' = (rel_year == -`k') * treated
    label variable pre_`k' "t-`k' (truoc chinh sach)"
}
* pre_1 bị bỏ làm reference (omitted)

forvalues k = 0(1)4 {
    gen post_`k' = (rel_year == `k') * treated
    label variable post_`k' "t+`k' (sau chinh sach)"
}

* Hồi quy event study
xtreg ln_gdp pre_4 pre_3 pre_2 ///
             post_0 post_1 post_2 post_3 post_4 ///
             trade_share edu infra i.year, ///
    fe vce(cluster province)

/*
   Đọc kết quả:
   - Hệ số pre_* (trước 2018): phải gần 0 và không có ý nghĩa
     → xác nhận Parallel Trends assumption (xu hướng song song)
   - Hệ số post_* (sau 2018): phải dương và tăng dần
     → tác động chính sách tích lũy theo thời gian
*/

* Vẽ đồ thị Event Study
coefplot, ///
    keep(pre_4 pre_3 pre_2 post_0 post_1 post_2 post_3 post_4) ///
    vertical yline(0) xline(3.5, lpattern(dash) lcolor(red)) ///
    title("Event Study – Kiem dinh Parallel Trends") ///
    xtitle("Nam tuong doi so voi nam chinh sach (t=0: nam 2018)") ///
    ytitle("He so (ATT uoc tinh)") ///
    xlabel(1 "t-4" 2 "t-3" 3 "t-2" 4 "t+0" 5 "t+1" 6 "t+2" 7 "t+3" 8 "t+4") ///
    note("Truoc 2018: nen gan 0 (Parallel Trends). Sau 2018: nen duong." ///
         "Duong do dung = nam bat dau chinh sach (2018). True ATT = 0.08")
graph export "figures/10_event_study_did.png", $png_opts
display "✓ figures/10_event_study_did.png"
/*
   Nếu coefplot chưa cài: ssc install coefplot
   Đồ thị này là bằng chứng thuyết phục nhất cho Parallel Trends
*/

drop pre_* post_* rel_year


/* ============================================================
   PHẦN 9: XUẤT KẾT QUẢ – ESTTAB & OUTREG2
   ============================================================ */

display _newline(2) "===== PHẦN 9: XUẤT KẾT QUẢ ====="

* Cài đặt packages nếu chưa có (bỏ comment để cài)
* ssc install estout,  replace
* ssc install outreg2, replace
* ssc install coefplot, replace

use "panel_buoi2.dta", clear
xtset province year

* --- Chạy lại các mô hình để lưu ---
eststo clear

eststo m1: quietly regress ln_gdp ln_fdi trade_share edu, ///
    vce(cluster province)

eststo m2: quietly regress ln_gdp ln_fdi trade_share edu i.year i.vung, ///
    vce(cluster province)

eststo m3: quietly xtreg ln_gdp ln_fdi trade_share edu, ///
    fe vce(cluster province)

eststo m4: quietly xtreg ln_gdp ln_fdi trade_share edu i.year, ///
    fe vce(cluster province)

eststo m5: quietly ivreg2 ln_gdp trade_share edu ///
    (ln_fdi = ln_dist elec_cost), cluster(province)

* --- 9.1 esttab (KHUYẾN NGHỊ) ---
display _newline "--- 9.1 esttab – xuat bang ket qua ---"

* Hiển thị trên màn hình
esttab m1 m2 m3 m4 m5, ///
    title("Bang 1: Tac dong FDI den Tang truong Kinh te Dia phuong") ///
    mtitles("OLS" "OLS+FE" "FE" "Two-way FE" "IV/2SLS") ///
    star(* 0.1 ** 0.05 *** 0.01) ///
    b(3) se(3) r2 ar2 ///
    scalars("N Quan sat") ///
    note("Ghi chu: SE chum theo tinh trong ngoac. True beta_FDI=0.35")

* Xuất sang file RTF (mở được trong Word)
esttab m1 m2 m3 m4 m5 using "bang_ketqua_chinh.rtf", ///
    title("Bang 1: Tac dong FDI den Tang truong Kinh te Dia phuong") ///
    mtitles("OLS" "OLS+FE" "FE" "Two-way FE" "IV/2SLS") ///
    star(* 0.1 ** 0.05 *** 0.01) ///
    b(3) se(3) r2 ar2 scalars("N") ///
    note("SE chum theo tinh trong ngoac. * p<0.1 ** p<0.05 *** p<0.01") ///
    replace
display "✓ Da xuat: bang_ketqua_chinh.rtf"

* Xuất sang LaTeX
esttab m1 m2 m3 m4 m5 using "bang_ketqua.tex", ///
    booktabs label replace ///
    star(* 0.1 ** 0.05 *** 0.01) b(3) se(3) r2 scalars("N")
display "✓ Da xuat: bang_ketqua.tex"

* --- 9.2 outreg2 (thêm mô hình lần lượt) ---
display _newline "--- 9.2 outreg2 – xuat sang Word ---"

* Mô hình 1: replace (tạo file mới)
quietly xtreg ln_gdp ln_fdi trade_share edu, fe vce(cluster province)
outreg2 using "bang_outreg2.doc", ///
    ctitle("FE co ban") dec(3) word replace ///
    addtext(Province FE, Yes, Year FE, No)

* Mô hình 2: append (thêm vào file đã có)
quietly xtreg ln_gdp ln_fdi trade_share edu i.year, fe vce(cluster province)
outreg2 using "bang_outreg2.doc", ///
    ctitle("Two-way FE") dec(3) word append ///
    addtext(Province FE, Yes, Year FE, Yes)

* Mô hình DiD
quietly xtreg ln_gdp did_var trade_share edu i.year, fe vce(cluster province)
outreg2 using "bang_outreg2.doc", ///
    ctitle("DiD (Two-way FE)") dec(3) word append ///
    addtext(Province FE, Yes, Year FE, Yes)

display "✓ Da xuat: bang_outreg2.doc"

* --- 9.3 Thống kê mô tả – bảng summary cho paper ---
display _newline "--- 9.3 Bang thong ke mo ta ---"
eststo clear
estpost summarize ln_gdp ln_fdi trade_share edu infra pop, detail
esttab using "bang_moata.rtf", ///
    cells("mean(fmt(3)) sd(fmt(3)) min(fmt(3)) max(fmt(3)) count(fmt(0))") ///
    title("Bang Thong ke Mo ta") ///
    collabels("Mean" "Std.Dev." "Min" "Max" "N") replace
display "✓ Da xuat: bang_moata.rtf"


/* ============================================================
   PHẦN 10: DO-FILE CHUYÊN NGHIỆP – BEST PRACTICES
   ============================================================ */

display _newline(2) "===== PHẦN 10: BEST PRACTICES ====="

display _newline "Cau truc do-file chuyen nghiep:"
display "  /project/"
display "    master.do           (goi cac do-file theo thu tu)"
display "    /data/"
display "      raw/              (du lieu goc – KHONG chinh sua)"
display "      clean/            (du lieu da xu ly)"
display "    /output/"
display "      tables/           (bang ket qua .rtf .tex)"
display "      figures/          (do thi .pdf .png)"
display "    01_clean_data.do"
display "    02_merge_data.do"
display "    03_analysis.do"
display "    04_tables.do"

/*
   MẪU HEADER CHUYÊN NGHIỆP cho do-file nghiên cứu:

   /*=================================================
     De tai: Tac dong FDI den Tang truong Kinh te DP
     Tac gia: NCS. Nguyen Van A – KTQT K28
     Giang vien HD: TS. Do Ngoc Kien
     Ngay tao:  2025-01-15  |  Lan sua cuoi: 2025-06-01
     Version:   3.2
     Du lieu:   63 tinh/TP VN, 2010-2022 (GSO)
     Phu thuoc: Stata 17 + estout + reghdfe + ivregress
   =================================================*/

   clear all
   set more off
   set maxvar 10000
   version 17.0

   * Global macros (khong hard-code duong dan tuyet doi)
   global root  "C:/Research/FDI_Project"
   global data  "$root/data"
   global out   "$root/output"

   * Ghi log toan phan
   log using "$out/run_`c(current_date)'.log", replace

   * Goi cac do-file
   do "$root/01_clean_data.do"
   do "$root/02_merge_data.do"
   do "$root/03_analysis.do"
   do "$root/04_tables.do"

   log close
*/

display _newline "Phim tat huu ich:"
display "  Ctrl+D       : Chay toan bo do-file"
display "  Ctrl+Shift+D : Chay phan duoc chon"
display "  Ctrl+/       : Comment / Uncomment dong"
display "  Ctrl+L       : Go to line"

display _newline "Packages nen cai (ssc install):"
display "  ssc install estout    // esttab, eststo"
display "  ssc install outreg2   // Xuat Word/LaTeX"
display "  ssc install coefplot  // Ve do thi he so"
display "  ssc install reghdfe   // OLS high-dimensional FE"
display "  ssc install ivreg2    // IV nang cao (thay ivregress)"
display "  ssc install csdid     // Staggered DiD C&S 2021"
display "  ssc install ietoolkit // Tools cho RCT/DiD"


/* ============================================================
   PHẦN 11: TỔNG KẾT – BẢNG TỔNG HỢP TẤT CẢ MÔ HÌNH
   ============================================================ */

display _newline(2) "===== PHẦN 11: TỔNG KẾT MÔ HÌNH ====="

use "panel_buoi2.dta", clear
xtset province year

eststo clear

eststo A_ols:   quietly regress  ln_gdp ln_fdi trade_share edu, ///
    vce(cluster province)

eststo B_ols_fe: quietly regress ln_gdp ln_fdi trade_share edu ///
    i.year i.vung, vce(cluster province)

eststo C_fe:    quietly xtreg   ln_gdp ln_fdi trade_share edu, ///
    fe vce(cluster province)

eststo D_fe2w:  quietly xtreg   ln_gdp ln_fdi trade_share edu ///
    i.year, fe vce(cluster province)

eststo E_iv:    quietly ivreg2 ln_gdp trade_share edu ///
    (ln_fdi = ln_dist elec_cost), cluster(province)

eststo F_did:   quietly xtreg   ln_gdp did_var trade_share edu ///
    i.year, fe vce(cluster province)

esttab A_ols B_ols_fe C_fe D_fe2w E_iv F_did, ///
    title("TONG KET: Tat ca mo hinh – Buoi 2") ///
    mtitles("OLS" "OLS+FE" "FE" "2-way FE" "IV" "DiD") ///
    star(* 0.1 ** 0.05 *** 0.01) b(3) se(3) r2 ///
    note("True beta_FDI=0.35 | True ATT(DiD)=0.08 | SE chum theo tinh")

esttab A_ols B_ols_fe C_fe D_fe2w E_iv F_did ///
    using "bang_tong_ket_buoi2.rtf", ///
    title("TONG KET: Buoi 2 – Phan tich Kinh te luong") ///
    mtitles("OLS" "OLS+FE" "FE" "2-way FE" "IV" "DiD") ///
    star(* 0.1 ** 0.05 *** 0.01) b(3) se(3) r2 scalars("N") ///
    note("True beta_FDI=0.35 | True ATT(DiD)=0.08") replace

display "✓ Da xuat: bang_tong_ket_buoi2.rtf"

* --- 9.4 Biểu đồ Coefplot so sánh hệ số các mô hình ---
display _newline "--- 9.4 Coefplot so sanh he so cac mo hinh ---"
coefplot A_ols B_ols_fe C_fe D_fe2w E_iv, ///
    keep(ln_fdi) ///
    asequation swapnames ///
    xline(0.35, lcolor(red) lpattern(dash)) ///
    title("He so FDI (ln_fdi) theo tung phuong phap uoc luong") ///
    xtitle("He so uoc luong + KTC 95%") ///
    note("Duong do dut = Gia tri that (beta=0.35 theo DGP mo phong)" ///
         "OLS bi chech len vi noi sinh. FE va IV gan voi gia tri that hon.")
graph export "figures/11_coefplot_compare_models.png", $png_opts
display "✓ figures/11_coefplot_compare_models.png"

* --- 9.5 Biểu đồ Parallel Trends trực quan (DiD) ---
display _newline "--- 9.5 Bieu do Parallel Trends truc quan ---"
use "panel_buoi2.dta", clear

preserve
collapse (mean) ln_gdp, by(year treated)
twoway (line ln_gdp year if treated==0, ///
            lcolor(navy) lwidth(medthick) lpattern(solid)) ///
       (line ln_gdp year if treated==1, ///
            lcolor(maroon) lwidth(medthick) lpattern(solid)) ///
       (line ln_gdp year if treated==0 & year>=2018, ///
            lcolor(navy) lwidth(medthick) lpattern(dash)) ///
       (line ln_gdp year if treated==1 & year>=2018, ///
            lcolor(maroon) lwidth(medthick) lpattern(dash)), ///
    xline(2017.5, lcolor(gs8) lpattern(dash) lwidth(thin)) ///
    legend(label(1 "Nhom kiem soat (truoc)") ///
           label(2 "Nhom xu ly (truoc)") ///
           label(3 "Nhom kiem soat (sau)") ///
           label(4 "Nhom xu ly (sau)") rows(2)) ///
    title("Parallel Trends – Difference-in-Differences") ///
    xtitle("Nam") ytitle("ln(GDP binh quan) trung binh") ///
    xlabel(2013(1)2022, angle(45)) ///
    note("Duong doc = Nam bat dau chinh sach (2018)." ///
         "Khoang cach gian dan giua 2 nhom sau 2018 = ATT uoc tinh")
graph export "figures/12_did_parallel_trends.png", $png_opts
display "✓ figures/12_did_parallel_trends.png"
restore

* --- 9.6 Biểu đồ phân phối Before/After theo nhóm (DiD visual) ---
twoway (kdensity ln_gdp if treated==1 & post==0, ///
            lcolor(navy) lpattern(dash) lwidth(medthick)) ///
       (kdensity ln_gdp if treated==1 & post==1, ///
            lcolor(navy) lpattern(solid) lwidth(medthick)) ///
       (kdensity ln_gdp if treated==0 & post==0, ///
            lcolor(maroon) lpattern(dash) lwidth(medthick)) ///
       (kdensity ln_gdp if treated==0 & post==1, ///
            lcolor(maroon) lpattern(solid) lwidth(medthick)), ///
    legend(label(1 "Xu ly – Truoc 2018") label(2 "Xu ly – Sau 2018") ///
           label(3 "Kiem soat – Truoc 2018") label(4 "Kiem soat – Sau 2018") ///
           rows(2) size(small)) ///
    title("Phan phoi GDP: Truoc va Sau chinh sach") ///
    xtitle("ln(GDP binh quan dau nguoi)") ytitle("Mat do") ///
    note("Nhom xu ly dich chuyen phai nhieu hon nhom kiem soat sau 2018")
graph export "figures/13_did_density_before_after.png", $png_opts
display "✓ figures/13_did_density_before_after.png"

* --- 9.7 Scatter IV: Stage 1 (FDI ~ Khoảng cách cảng) ---
display _newline "--- 9.7 Bieu do Stage 1 (IV relevance) ---"
twoway (scatter ln_fdi ln_dist, mcolor(navy%50) msize(vsmall)) ///
       (lfit    ln_fdi ln_dist, lcolor(red) lwidth(medthick)), ///
    title("Stage 1 IV: FDI va Khoang cach den Cang bien") ///
    xtitle("ln(Khoang cach den cang bien)") ytitle("ln(FDI + 1)") ///
    legend(label(1 "Quan sat") label(2 "Duong OLS fit")) ///
    note("Bien cong cu hop le: tuong quan am voi FDI (tinh xa cang it FDI hon)")
graph export "figures/14_iv_stage1_scatter.png", $png_opts
display "✓ figures/14_iv_stage1_scatter.png"

xtset province year

* Đóng log
display _newline "=== Ket thuc Buoi 2: $S_DATE $S_TIME ==="
log close


/* ============================================================
   GHI CHÚ CHO HỌC VIÊN
   ============================================================

   FILE TẠO RA SAU KHI CHẠY DO-FILE NÀY:

   DỮ LIỆU:
   ✓ panel_buoi2.dta            – Dataset mô phỏng chính (630 obs)

   BẢNG KẾT QUẢ:
   ✓ bang_ketqua_chinh.rtf      – Bảng kết quả chính (5 mô hình)
   ✓ bang_ketqua.tex            – Bảng LaTeX (journal-ready)
   ✓ bang_outreg2.doc           – Bảng Word (outreg2)
   ✓ bang_moata.rtf             – Bảng thống kê mô tả
   ✓ bang_tong_ket_buoi2.rtf    – Bảng tổng kết tất cả mô hình

   BIỂU ĐỒ PNG (trong thư mục figures/):
   ✓ 01_histogram_lngdp.png     – Phân phối ln(GDP)
   ✓ 02_boxplot_gdp_vung.png    – Box plot GDP theo vùng
   ✓ 03_trend_gdp_by_vung.png   – Xu hướng GDP 2013–2022 theo vùng
   ✓ 04_scatter_fdi_gdp.png     – Scatter FDI vs GDP (màu theo vùng)
   ✓ 05_histogram_resid_ols.png – Phân phối phần dư OLS
   ✓ 06_pnorm_resid_ols.png     – P-P plot phần dư (chuẩn tắc)
   ✓ 07_rvfplot_ols.png         – Residual vs Fitted plot
   ✓ 08_margins_fdi_by_vung.png – Hiệu ứng biên FDI theo vùng
   ✓ 09_margins_fdi_by_infra.png– Hiệu ứng biên FDI theo hạ tầng
   ✓ 10_event_study_did.png     – Event Study (Parallel Trends check)
   ✓ 11_coefplot_compare_models.png – So sánh beta_FDI các mô hình
   ✓ 12_did_parallel_trends.png – Parallel Trends trực quan
   ✓ 13_did_density_before_after.png – Phân phối Before/After DiD
   ✓ 14_iv_stage1_scatter.png   – Stage 1 IV: FDI ~ Khoảng cách cảng

   LOG:
   ✓ buoi2_stata_mau.log        – Log toàn bộ phiên làm việc

   TRUE DGP (để học viên so sánh kết quả ước lượng):
   - beta_FDI (ln_fdi → ln_gdp) = 0.35
   - ATT chính sách DiD          = 0.08 (8%)
   - FDI NỘI SINH: tương quan với province FE → OLS bị chệch lên

   THÔNG ĐIỆP CHÍNH:
   ① OLS > IV > FE: hệ số beta_FDI giảm dần khi kiểm soát bias tốt hơn
   ② Hausman p<0.05 → FE tốt hơn RE khi FDI nội sinh với alpha_i
   ③ Event study: hệ số pre_* ≈ 0 → Parallel Trends được thỏa mãn
   ④ esttab tạo bảng publication-ready trong 1 dòng lệnh

   YÊU CẦU PACKAGES (cài 1 lần trước khi chạy):
   ssc install estout
   ssc install outreg2
   ssc install coefplot
   ssc install ivreg2

   CÁCH XUẤT PNG – GHI NHỚ:
   - graph export "ten_file.png", as(png) width(1600) height(1200) replace
   - width/height tính bằng pixel (1600×1200 = Full HD, phù hợp paper)
   - Thay "png" bằng "pdf" để xuất PDF vector (chất lượng cao hơn cho LaTeX)
   - Thay "png" bằng "eps" nếu tạp chí yêu cầu định dạng EPS

   ============================================================ */

* ===== KẾT THÚC DO-FILE BUỔI 2 =====