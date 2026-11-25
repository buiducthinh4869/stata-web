*===============================================================================
* HỌC PHẦN: ĐÁNH GIÁ TÁC ĐỘNG CHÍNH SÁCH (Quantitative Policy Impact Evaluation)
* CHƯƠNG TRÌNH TIẾN SĨ QUẢN LÝ KINH TẾ - TS QLKT 1B - FTU
*
* BUỔI 4: Randomized Promotion Trial (Encouragement Design) & Instrumental Variables
*
* Tác giả: TS. Đỗ Ngọc Kiên
* Ngày:    13/05/2026
*
* Mục tiêu của do-file:
*   1. Mô tả dataset rpt_smetraining.dta (mô phỏng 2.000 DNNVV Việt Nam)
*   2. Chạy OLS naive và OLS có kiểm soát → minh hoạ thiên lệch nội sinh
*   3. Kiểm tra cân bằng của instrument Z
*   4. First stage rõ ràng → kiểm định Relevance
*   5. Reduced form (ITT)
*   6. 2SLS chính → ước lượng LATE
*   7. Weak-IV diagnostics (Olea-Pflueger, Anderson-Rubin)
*   8. So sánh OLS vs IV → diễn giải chiều thiên lệch
*   9. Xuất bảng kết quả cho báo cáo
*  10. Kiểm chứng đẳng thức Wald
*  11. Trực quan hoá kết quả (biểu đồ)
*  12. Bài tập mở rộng
*
* Yêu cầu: Stata 15+ với các package: ivreg2, ranktest, weakivtest, estout, coefplot
*
* File xuất ra:
*   bieu_do_1_noi_sinh.png          — Nguồn gốc nội sinh (U_unobs vs lnY)
*   bieu_do_2_ols_naive.png         — OLS naive scatter + fitted line
*   bieu_do_3_balance_test.png      — Coefplot balance test Z
*   bieu_do_4_takeup_compliance.png — Compliance gap bar chart
*   bieu_do_5_so_sanh_ols_iv.png    — So sánh hệ số M1–M5 (QUAN TRỌNG NHẤT)
*   bieu_do_6_wald_decomposition.png— Phân rã Wald estimator (2 panel)
*   bang_1_ket_qua_chinh.rtf / .tex — Bảng kết quả chính
*   bang_2_first_stage.rtf / .tex   — Bảng First stage + diagnostics
*===============================================================================

*-------------------------------------------------------------------------------
* 0. Setup
*-------------------------------------------------------------------------------
version 15
clear all
set more off

* (Chạy 1 lần) Cài package nếu chưa có
* ssc install ivreg2,     replace
* ssc install ranktest,   replace
* ssc install weakivtest, replace
* ssc install estout,     replace
* ssc install coefplot,   replace

* Đặt thư mục làm việc — SỬA cd cho phù hợp máy của bạn
* cd "C:/Users/<your_path>/Buoi4_RPT_IV"

* Thiết lập scheme đồ thị nhất quán
set scheme s2color

*-------------------------------------------------------------------------------
* 1. Mở dữ liệu và mô tả
*-------------------------------------------------------------------------------
use "rpt_smetraining.dta", clear

* Kiểm tra cấu trúc dữ liệu
describe

* Thống kê mô tả
summarize firm_id Z D lnY size_pre age sector region, sep(0)

* Phân phối Z (kỳ vọng ~50/50 vì ngẫu nhiên hoàn hảo)
tabulate Z, missing

* Take-up theo nhóm Z (compliance)
tabulate Z D, row

* Take-up theo từng ngành — kiểm tra heterogeneity
tabulate sector D if Z==1, row
tabulate sector D if Z==0, row

*-------------------------------------------------------------------------------
* 2. OLS NAIVE — kỳ vọng thiên lệch lên do self-selection
*-------------------------------------------------------------------------------
* Spec 1: OLS không kiểm soát
regress lnY D, vce(robust)
estimates store M1_OLS_naive

* Spec 2: OLS có kiểm soát biến tiền can thiệp
regress lnY D size_pre age i.sector i.region, vce(robust)
estimates store M2_OLS_X

* Kỳ vọng: β̂_OLS > β thật (0.15) vì U dương tương quan với cả D và Y
* Trong dataset mô phỏng: β̂_OLS_naive ≈ 0.36, β̂_OLS_X ≈ 0.31

*-------------------------------------------------------------------------------
* BIỂU ĐỒ 1: Scatter lnY vs U_unobs phân theo D
*   Mục đích: cho NCS thấy nhóm D=1 có U cao hơn → self-selection → OLS bias
*   Lưu ý: chỉ khả dụng trong dataset mô phỏng vì U thực tế không quan sát được
*-------------------------------------------------------------------------------
twoway ///
    (scatter lnY U_unobs if D==0, mcolor(navy%40)     msymbol(Oh) msize(vsmall) ///
        legend(label(1 "Không tham gia (D=0)"))) ///
    (scatter lnY U_unobs if D==1, mcolor(cranberry%40) msymbol(O)  msize(vsmall) ///
        legend(label(2 "Tham gia (D=1)"))) ///
    (lfit lnY U_unobs if D==0, lcolor(navy)      lwidth(medthick) legend(label(3 "Xu hướng D=0"))) ///
    (lfit lnY U_unobs if D==1, lcolor(cranberry)  lwidth(medthick) legend(label(4 "Xu hướng D=1"))), ///
    title("Biểu đồ 1: Nhóm tham gia có năng lực nền cao hơn") ///
    subtitle("Minh hoạ nguồn gốc nội sinh — chỉ thấy được trong dữ liệu mô phỏng") ///
    xtitle("Năng lực không quan sát được (U_unobs)") ///
    ytitle("Logarithm doanh thu (lnY)") ///
    note("Ghi chú: Nhóm D=1 tập trung ở vùng U cao → OLS đang đo lẫn" ///
         "năng lực nền vào tác động đào tạo → overestimate.", size(vsmall)) ///
    legend(order(1 2) rows(1)) ///
    graphregion(color(white)) bgcolor(white)
graph export "bieu_do_1_noi_sinh.png", replace width(1200)

*-------------------------------------------------------------------------------
* BIỂU ĐỒ 2: Scatter lnY vs D + OLS fitted line
*   Mục đích: hình ảnh hoá hệ số OLS naive, đường dốc hơn β thật rõ ràng
*-------------------------------------------------------------------------------
twoway ///
    (scatter lnY D, mcolor(navy%25) msymbol(Oh) msize(vsmall) jitter(3) ///
        legend(label(1 "Quan sát (lnY, D)"))) ///
    (lfit lnY D, lcolor(cranberry) lwidth(thick) ///
        legend(label(2 "OLS fit (β̂ ≈ 0.36)"))), ///
    title("Biểu đồ 2: OLS naive — thiên lệch do tự chọn") ///
    subtitle("Hệ số OLS ≈ 0.36 so với β thật = 0.15") ///
    xtitle("Tham gia đào tạo (D: 0 = Không, 1 = Có)") ///
    ytitle("Logarithm doanh thu (lnY)") ///
    xlabel(0(0.2)1) ///
    note("Ghi chú: Đường OLS dốc gấp đôi β thật vì nhóm D=1 có năng lực nền cao hơn.", ///
        size(vsmall)) ///
    legend(order(2) rows(1)) ///
    graphregion(color(white)) bgcolor(white)
graph export "bieu_do_2_ols_naive.png", replace width(1200)

*-------------------------------------------------------------------------------
* 3. KIỂM TRA CÂN BẰNG của Z (balance test)
*    Mục tiêu: kiểm tra giả định Independence (Z ⊥ X)
*-------------------------------------------------------------------------------
* Bảng cân bằng đơn lẻ
foreach v of varlist size_pre age {
    display _newline as text "*** Biến: `v' ***"
    regress `v' Z, vce(robust)
}

* F-joint test: hồi quy Z theo toàn bộ X
regress Z size_pre age i.sector i.region, vce(robust)
testparm size_pre age i.sector i.region
display "F-joint test cân bằng: nếu p-value > 0.1 → OK"

* Lưu kết quả để vẽ coefplot
quietly regress size_pre Z, vce(robust)
estimates store bal_size
quietly regress age Z, vce(robust)
estimates store bal_age
quietly regress D Z, vce(robust)
estimates store bal_D

* Lưu ý: balance test chỉ kiểm tra biến QUAN SÁT ĐƯỢC.

*-------------------------------------------------------------------------------
* BIỂU ĐỒ 3: Coefplot balance test + first stage trực quan
*   Mục đích: một nhìn thấy Z không tương quan với X (size_pre, age)
*             nhưng tương quan mạnh với D → instrument hợp lệ về relevance
*-------------------------------------------------------------------------------
coefplot ///
    (bal_size, label("Quy mô DN (size_pre)")              ///
        msymbol(D) mcolor(navy)     ciopt(lcolor(navy)     lwidth(medthick))) ///
    (bal_age,  label("Tuổi doanh nghiệp (age)")           ///
        msymbol(S) mcolor(dkorange) ciopt(lcolor(dkorange) lwidth(medthick))) ///
    (bal_D,    label("Tham gia đào tạo (D) — First Stage") ///
        msymbol(O) mcolor(cranberry) ciopt(lcolor(cranberry) lwidth(medthick))), ///
    vertical drop(_cons) ///
    yline(0, lcolor(black) lpattern(dash) lwidth(thin)) ///
    title("Biểu đồ 3: Kiểm tra cân bằng của Instrument Z") ///
    subtitle("Hệ số hồi quy của Z lên từng biến, khoảng tin cậy 95%") ///
    ytitle("Hệ số hồi quy (OLS)") ///
    note("Ghi chú: size_pre và age không khác 0 → Z cân bằng (Independence đạt)." ///
         "D khác 0 rõ ràng → First stage mạnh (Relevance đạt).", size(vsmall)) ///
    xlabel(, angle(0)) ///
    graphregion(color(white)) bgcolor(white)
graph export "bieu_do_3_balance_test.png", replace width(1200)

*-------------------------------------------------------------------------------
* 4. FIRST STAGE — kiểm định Relevance: Cov(Z, D) ≠ 0
*-------------------------------------------------------------------------------
regress D Z size_pre age i.sector i.region, vce(robust)
test Z

* Kỳ vọng: F-statistic của Z lớn (kỳ vọng > 100 trong dataset này)
* Quy tắc: F > 10 (Staiger-Stock); F_eff > 23.1 (Olea-Pflueger)

estimates store M3_FS

display _newline "Take-up theo Z:"
mean D, over(Z)

*-------------------------------------------------------------------------------
* BIỂU ĐỒ 4: Bar chart take-up rate theo Z — compliance gap
*   Mục đích: NCS thấy Z tạo ra sự chênh lệch lớn trong P(D=1)
*             Khoảng cách giữa hai cột chính là First stage
*-------------------------------------------------------------------------------
quietly mean D if Z==0
scalar d_z0 = r(table)[1,1]
quietly mean D if Z==1
scalar d_z1 = r(table)[1,1]

preserve
    clear
    set obs 2
    gen nhom_z  = _n - 1
    gen takeup  = cond(nhom_z==0, scalar(d_z0), scalar(d_z1))

    twoway ///
        (bar takeup nhom_z, barwidth(0.5) ///
            color(navy%70 cranberry%70) ///
            ylabel(0(0.1)1, format(%3.1f))), ///
        title("Biểu đồ 4: Tỷ lệ tham gia theo nhóm Instrument Z") ///
        subtitle("Compliance gap = P(D=1|Z=1) - P(D=1|Z=0) = First Stage") ///
        ytitle("Tỷ lệ tham gia đào tạo P(D=1)") ///
        xtitle("") ///
        xlabel(0 `"Không được mời (Z=0)"' 1 `"Được mời tham gia (Z=1)"', noticks) ///
        note("Ghi chú: Khoảng cách giữa hai cột = First stage." ///
             "Cột phải cao hơn rõ rệt → Instrument đủ mạnh.", size(vsmall)) ///
        legend(off) ///
        graphregion(color(white)) bgcolor(white)
    graph export "bieu_do_4_takeup_compliance.png", replace width(1000)
restore

*-------------------------------------------------------------------------------
* 5. REDUCED FORM (ITT) — hồi quy Y theo Z
*-------------------------------------------------------------------------------
regress lnY Z size_pre age i.sector i.region, vce(robust)
estimates store M4_ITT

* Kỳ vọng: hệ số Z ≈ 0.05 (ITT nhỏ vì bị pha loãng bởi non-compliance)

*-------------------------------------------------------------------------------
* 6. 2SLS — ước lượng IV chính thức
*-------------------------------------------------------------------------------
ivreg2 lnY (D = Z) size_pre age i.sector i.region, robust first
estimates store M5_IV

* Diễn giải:
*   - Coefficient on D = LATE (kỳ vọng ≈ 0.15-0.18, gần β thật)
*   - First-stage F — phải > 10 (Staiger-Stock) hoặc > 23.1 (Olea-Pflueger)
*   - Stock-Yogo / Kleibergen-Paap rk Wald F (báo cáo trong ivreg2)

*-------------------------------------------------------------------------------
* 7. WEAK-IV DIAGNOSTICS bổ sung
*-------------------------------------------------------------------------------
* Olea-Pflueger effective F (cần ssc install weakivtest)
weakivtest
* So sánh với critical value 23.1 (5% maximum relative bias)

* Anderson-Rubin CI
ivreg2 lnY (D = Z) size_pre age i.sector i.region, robust
* Hoặc: rivtest, ci(ar) — nếu đã cài rivtest

*-------------------------------------------------------------------------------
* 8. SO SÁNH OLS vs IV — bảng tổng hợp console
*-------------------------------------------------------------------------------
* Xoá Mata objects tạm để tránh conflict khi chạy lại trong cùng session
mata: mata clear

estimates table M1_OLS_naive M2_OLS_X M3_FS M4_ITT M5_IV, ///
    keep(D Z) b(%9.4f) se(%9.4f) stats(N r2)

* Diễn giải kỳ vọng:
*   M1 OLS naive:     β̂ ≈ 0.36 (bias dương lớn)
*   M2 OLS + X:       β̂ ≈ 0.31 (bias giảm nhưng còn)
*   M3 First stage Z: π̂ ≈ 0.35 (compliance gap mạnh, F lớn)
*   M4 ITT:           γ̂_Z ≈ 0.05
*   M5 2SLS LATE:     β̂ ≈ 0.15-0.18 (gần β thật)

* Hausman test so sánh OLS và IV
quietly regress lnY D size_pre age i.sector i.region, vce(robust)
estimates store ols_for_hausman
quietly ivreg2 lnY (D = Z) size_pre age i.sector i.region, robust
* hausman ols_for_hausman M5_IV, sigmamore   // bỏ comment nếu cần

*-------------------------------------------------------------------------------
* BIỂU ĐỒ 5: Coefplot so sánh hệ số D/Z qua M1→M5
*   Mục đích: đây là biểu đồ quan trọng nhất — NCS thấy bias giảm dần
*             từ OLS naive → OLS+X → 2SLS, và 2SLS hội tụ về β thật
*-------------------------------------------------------------------------------
coefplot ///
    (M1_OLS_naive, label("(1) OLS Naive")           ///
        msymbol(O) msize(large) mcolor(cranberry)    ///
        ciopt(lcolor(cranberry) lwidth(medthick))) ///
    (M2_OLS_X,     label("(2) OLS + Kiểm soát X")   ///
        msymbol(D) msize(large) mcolor(dkorange)     ///
        ciopt(lcolor(dkorange) lwidth(medthick))) ///
    (M4_ITT,       label("(3) Reduced Form (ITT)")   ///
        msymbol(S) msize(large) mcolor(dkgreen)      ///
        ciopt(lcolor(dkgreen) lwidth(medthick))) ///
    (M5_IV,        label("(4) 2SLS — LATE")          ///
        msymbol(T) msize(large) mcolor(navy)         ///
        ciopt(lcolor(navy) lwidth(medthick))), ///
    keep(D Z) ///
    xline(0.15, lcolor(red) lpattern(dash) lwidth(medthick)) ///
    title("Biểu đồ 5: So sánh ước lượng OLS và 2SLS") ///
    subtitle("Hệ số tác động của D (và Z trong ITT) với khoảng tin cậy 95%") ///
    xtitle("Ước lượng hệ số và khoảng tin cậy 95%") ///
    note("Đường đỏ đứt = β thật = 0.15 (biết được vì dữ liệu mô phỏng)." ///
         "OLS overestimate rõ; 2SLS hội tụ về β thật.", size(vsmall)) ///
    legend(rows(2) size(small)) ///
    graphregion(color(white)) bgcolor(white)
graph export "bieu_do_5_so_sanh_ols_iv.png", replace width(1300)

*-------------------------------------------------------------------------------
* 9. XUẤT BẢNG KẾT QUẢ ra RTF / LaTeX
*-------------------------------------------------------------------------------

* --- Bảng 1: Kết quả chính OLS và 2SLS ---
esttab M1_OLS_naive M2_OLS_X M4_ITT M5_IV using bang_1_ket_qua_chinh.rtf, replace ///
    keep(D Z) b(%9.3f) se(%9.3f) ///
    star(* 0.10 ** 0.05 *** 0.01) ///
    nonotes addnotes("Sai số chuẩn bền vững heteroskedasticity trong ngoặc." ///
                     "* p<0.10, ** p<0.05, *** p<0.01." ///
                     "Mẫu: 2.000 DNNVV mô phỏng. Kiểm soát: quy mô, tuổi DN, ngành, vùng.") ///
    mtitles("OLS Naive" "OLS + X" "Reduced Form" "2SLS") ///
    title("Bảng 1: So sánh OLS, Reduced Form và 2SLS") ///
    scalars("N Số quan sát" "r2 R-squared")

* --- Bảng 2: First stage và weak-IV diagnostics ---
esttab M3_FS using bang_2_first_stage.rtf, replace ///
    keep(Z) b(%9.3f) se(%9.3f) ///
    star(* 0.10 ** 0.05 *** 0.01) ///
    nonotes addnotes("Biến phụ thuộc: D (tham gia đào tạo)." ///
                     "Hệ số Z = compliance gap = First stage." ///
                     "F > 10 (Staiger-Stock); F_eff > 23.1 (Olea-Pflueger) → instrument mạnh." ///
                     "* p<0.10, ** p<0.05, *** p<0.01.") ///
    mtitles("First Stage") ///
    title("Bảng 2: First Stage — Kiểm định Relevance của Instrument Z") ///
    scalars("N Số quan sát" "r2 R-squared")

* --- LaTeX versions ---
esttab M1_OLS_naive M2_OLS_X M4_ITT M5_IV using bang_1_ket_qua_chinh.tex, replace ///
    keep(D Z) b(%9.3f) se(%9.3f) ///
    star(* 0.10 ** 0.05 *** 0.01) ///
    booktabs nonotes addnotes("Robust SE in parentheses. N=2000 simulated SMEs." ///
                               "Controls: firm size, age, sector FE, region FE.") ///
    mtitles("OLS Naive" "OLS + X" "Reduced Form" "2SLS") ///
    title("Bảng 1: So sánh OLS, Reduced Form và 2SLS")

esttab M3_FS using bang_2_first_stage.tex, replace ///
    keep(Z) b(%9.3f) se(%9.3f) ///
    star(* 0.10 ** 0.05 *** 0.01) ///
    booktabs nonotes addnotes("Dependent variable: D. Robust SE." ///
                               "F > 10 (Staiger-Stock); F_eff > 23.1 (Olea-Pflueger).") ///
    mtitles("First Stage") ///
    title("Bảng 2: First Stage — Instrument Relevance")

*-------------------------------------------------------------------------------
* 10. KIỂM CHỨNG ĐẲNG THỨC WALD bằng tay (minh hoạ tại lớp)
*-------------------------------------------------------------------------------
quietly mean lnY if Z==1
matrix m1 = r(table)
scalar yz1 = m1[1,1]
quietly mean lnY if Z==0
matrix m2 = r(table)
scalar yz0 = m2[1,1]

quietly mean D if Z==1
matrix m3 = r(table)
scalar dz1 = m3[1,1]
quietly mean D if Z==0
matrix m4 = r(table)
scalar dz0 = m4[1,1]

display _newline "=== Kiểm chứng Wald estimator (không kiểm soát X) ==="
display "E[Y|Z=1]    = " %9.4f yz1
display "E[Y|Z=0]    = " %9.4f yz0
display "ITT          = " %9.4f (yz1 - yz0)
display "P(D=1|Z=1)  = " %9.4f dz1
display "P(D=1|Z=0)  = " %9.4f dz0
display "First stage  = " %9.4f (dz1 - dz0)
display "Wald LATE    = " %9.4f ((yz1 - yz0) / (dz1 - dz0))

*-------------------------------------------------------------------------------
* BIỂU ĐỒ 6: Bar chart 2 panel — phân rã Wald estimator
*   Mục đích: trực quan hoá phép chia LATE = ITT / First Stage
*   Panel A: E[Y|Z=1] vs E[Y|Z=0] → khoảng cách = ITT (tử số)
*   Panel B: P(D=1|Z=1) vs P(D=1|Z=0) → khoảng cách = First stage (mẫu số)
*-------------------------------------------------------------------------------
preserve
    clear
    set obs 4
    gen panel   = cond(_n <= 2, 1, 2)
    gen nhom    = mod(_n - 1, 2)                // 0 hoặc 1

    gen gia_tri = .
    replace gia_tri = scalar(yz0) if _n == 1
    replace gia_tri = scalar(yz1) if _n == 2
    replace gia_tri = scalar(dz0) if _n == 3
    replace gia_tri = scalar(dz1) if _n == 4

    * Panel A: E[Y|Z]
    twoway ///
        (bar gia_tri nhom if panel==1, barwidth(0.5) color(navy%75) ///
            ylabel(, format(%5.2f))), ///
        title("Panel A: E[Y|Z]") ///
        subtitle("Tử số của Wald") ///
        ytitle("E[lnY]") xtitle("") ///
        xlabel(0 `"Z=0"' 1 `"Z=1"', noticks) ///
        note("ITT = E[Y|Z=1] - E[Y|Z=0]") ///
        legend(off) name(p_a, replace) ///
        graphregion(color(white)) bgcolor(white)

    * Panel B: P(D=1|Z)
    twoway ///
        (bar gia_tri nhom if panel==2, barwidth(0.5) color(cranberry%75) ///
            ylabel(0(0.2)1, format(%3.1f))), ///
        title("Panel B: P(D=1|Z)") ///
        subtitle("Mẫu số của Wald") ///
        ytitle("P(D=1)") xtitle("") ///
        xlabel(0 `"Z=0"' 1 `"Z=1"', noticks) ///
        note("First Stage = P(D=1|Z=1) - P(D=1|Z=0)") ///
        legend(off) name(p_b, replace) ///
        graphregion(color(white)) bgcolor(white)

    graph combine p_a p_b, ///
        title("Biểu đồ 6: Phân rã Wald Estimator") ///
        subtitle("LATE = Khoảng cách Panel A / Khoảng cách Panel B") ///
        note("Ghi chú: Phép chia Wald loại bỏ hiệu ứng pha loãng của non-compliance." ///
             "Chỉ đo tác động trên compliers — nhóm thay đổi D do Z.", size(vsmall)) ///
        rows(1) graphregion(color(white))
    graph export "bieu_do_6_wald_decomposition.png", replace width(1400)
restore

*-------------------------------------------------------------------------------
* 11. BÀI TẬP MỞ RỘNG (tự thực hành)
*-------------------------------------------------------------------------------
* (a) Heterogeneous LATE theo ngành:
*     foreach k of numlist 1/4 {
*         display "Ngành = `k'"
*         ivreg2 lnY (D = Z) size_pre age if sector==`k', robust
*     }
*
* (b) Đặc điểm compliers (Abadie 2003 kappa-weighted means):
*     gen kappa = 1 - (D * (1-Z)) / (1 - mean(Z)) - ((1-D) * Z) / mean(Z)
*     summarize size_pre [aweight=kappa]   // compliers' mean of size_pre
*
* (c) Placebo test: hồi quy outcome không liên quan (tuổi DN) theo Z
*     reg age Z, vce(robust)  // kỳ vọng coefficient ~0
*
* (d) Sensitivity test với DGP khác: thay đổi seed Python và xem β̂_IV có ổn không

*-------------------------------------------------------------------------------
* GHI CHÚ CHO NCS:
*   1. Biểu đồ 1 dùng U_unobs — chỉ tồn tại trong dữ liệu mô phỏng để minh hoạ
*      tại lớp. Trong nghiên cứu thực tế, KHÔNG có biến này.
*   2. Khi báo cáo IV trong luận án, LUÔN báo cáo đủ:
*      (i) OLS naive + OLS có kiểm soát
*      (ii) First stage F-statistic (Staiger-Stock và Olea-Pflueger)
*      (iii) Reduced form
*      (iv) 2SLS với weak-IV CI (Anderson-Rubin hoặc rivtest)
*      (v) Diễn giải LATE ≠ ATE và đặc điểm của compliers
*   3. Tham khảo: Angrist & Pischke (2009) Mostly Harmless Econometrics, Ch. 4.
*===============================================================================
* End of do-file
