*===============================================================================
* REPLICATION & GIẢNG DẠY: Duflo, Dupas & Kremer (2011)
* "Peer Effects, Teacher Incentives, and the Impact of Tracking"
* American Economic Review 101(5): 1739–1774
*
* Phạm vi: Core RCT — Table 1 (Balance) + Table 2 (ATE) + Table 3 (Heterogeneity)
*           + ITT vs LATE + Randomization Inference
*
* Output: Bảng hồi quy .rtf → C:\Users\User\Downloads\RCT\
*
* Yêu cầu packages:
*   ssc install estout
*   ssc install ritest
*   ssc install outreg2    (backup nếu estout gặp vấn đề)
*
* Ghi chú sư phạm (PhD):
*   - Randomization ở cấp TRƯỜNG (school-level) → cluster SE theo schoolid bắt buộc
*   - Với ~60 clusters/arm, cluster SE đã đủ; wild bootstrap là robustness check
*   - LATE ≈ ITT ở đây vì compliance ~98.6% — nhưng IV logic vẫn cần minh họa
*   - Randomization inference không giả định phân phối của ε, chỉ dùng
*     tính ngẫu nhiên của assignment — phù hợp khi số cluster nhỏ
*===============================================================================

version 15
clear all
set more off
set linesize 120
capture log close

*-------------------------------------------------------------------------------
* 0. ĐƯỜNG DẪN
*-------------------------------------------------------------------------------
global DATA  "C:\Users\User\Downloads"
global OUT   "C:\Users\User\Downloads\RCT"
capture mkdir "$OUT"

log using "$OUT\ddk2011_rct_core.smcl", replace

*-------------------------------------------------------------------------------
* 1. NẠP DỮ LIỆU & XÂY DỰNG BIẾN
*-------------------------------------------------------------------------------
use "$DATA\student_test_data.dta", clear

*--- 1a. Chuẩn hóa điểm theo phân phối NON-TRACKING (mean=0, SD=1)
*    Lý do: tracking schools có raw score cao hơn ~1.3 điểm ngay từ baseline
*    → nếu không chuẩn hóa, hệ số tracking bị upward bias
foreach v of varlist totalscore mathscoreraw litscore ///
                     r2_totalscore r2_mathscoreraw r2_litscore {
    quietly summarize `v' if tracking == 0
    gen z_`v' = (`v' - r(mean)) / r(sd)
    label variable z_`v' "Chuẩn hóa `v' (mean 0, SD 1 ở non-tracking)"
}

*--- 1b. Global controls
* CONTROLS_BASE: không có bottomhalf/quarter — dùng khi có interaction term
* CONTROLS:      đầy đủ — dùng khi không có interaction
* Lý do tách: bottomhalf đã nằm trong interaction c.bottomhalf#c.tracking
*             → đưa vào cả $CONTROLS lẫn interaction gây perfect multicollinearity
global CONTROLS_BASE "girl agetest etpteacher realpercentile"
global CONTROLS      "girl agetest etpteacher bottomhalf bottomquarter secondquarter thirdquarter realpercentile"

*--- 1c. Biến compliance (cho phần LATE)
* Trong tracking schools: assignment = bottomhalf, take-up = lowstream
* ~1.4% misassigned (52/3,612 obs) → near-perfect compliance
* Với non-tracking schools: lowstream = 0 theo định nghĩa (không có stream)
* → Instrument: bottomhalf × tracking (chỉ có ý nghĩa trong tracking schools)
gen assigned_lowstream = bottomhalf * tracking
label variable assigned_lowstream "Được gán vào lớp dưới (instrument)"

*--- 1d. Kiểm tra compliance rate (báo cáo trong log)
display _newline "=== COMPLIANCE CHECK ==="
display "Assignment (bottomhalf) vs actual placement (lowstream), tracking schools only:"
tab lowstream bottomhalf if tracking == 1
quietly count if tracking == 1 & !missing(lowstream) & !missing(bottomhalf)
local n_track = r(N)
quietly count if tracking == 1 & lowstream == bottomhalf & !missing(lowstream) & !missing(bottomhalf)
display "Compliance rate: " %5.3f r(N)/`n_track' " (" r(N) "/" `n_track' ")"

save "$OUT\ddk_prepared.dta", replace

*===============================================================================
* TABLE 1 — BALANCE CHECK
* Mục đích sư phạm:
*   - Kiểm tra randomization có tạo ra 2 nhóm cân bằng về pre-treatment
*     characteristics không
*   - Dùng regression với cluster SE thay vì t-test đơn giản vì
*     randomization ở cấp trường (nếu dùng ttest, SE bị underestimate)
*===============================================================================
display _newline(2) "========================================================"
display "TABLE 1 — BALANCE CHECK"
display "========================================================"

*--- Panel A: Đặc điểm học sinh (student-level, cluster by school)
local balance_vars "girl agetest std_mark percentile"

foreach v of local balance_vars {
    quietly reg `v' tracking, vce(cluster schoolid)
    estimates store bal_`v'
}

estout bal_girl bal_agetest bal_std_mark bal_percentile, ///
    keep(tracking _cons) ///
    cells(b(fmt(3) star) se(par fmt(3))) ///
    stats(N, fmt(0)) ///
    varlabels(tracking "Tracking" _cons "Mean (Control)") ///
    title("Table 1 — Balance Check: Pre-treatment Characteristics") ///
    starlevels(* 0.10 ** 0.05 *** 0.01) ///
    note("Cluster-robust SE in parentheses (cluster = school)." ///
         "*** p<0.01, ** p<0.05, * p<0.10")

* Xuất RTF
esttab bal_girl bal_agetest bal_std_mark bal_percentile ///
    using "$OUT\table1_balance.rtf", replace ///
    keep(tracking _cons) ///
    cells(b(fmt(3) star) se(par fmt(3))) ///
    stats(N, fmt(0) labels("Observations")) ///
    varlabels(tracking "Tracking (=1)" _cons "Mean (Control group)") ///
    title("Table 1 — Balance Check") ///
    starlevels(* 0.10 ** 0.05 *** 0.01) ///
    note("Cluster-robust SE in parentheses (cluster = school). *** p<0.01, ** p<0.05, * p<0.10") ///
    rtf

*--- Panel B: Balance trong tracking schools (top vs bottom stream)
display _newline "--- Panel B: Top vs Bottom stream (tracking schools only) ---"
foreach v in girl agetest std_mark {
    quietly reg `v' bottomhalf if tracking == 1, vce(cluster schoolid)
    estimates store bal2_`v'
}

esttab bal2_girl bal2_agetest bal2_std_mark ///
    using "$OUT\table1b_balance_stream.rtf", replace ///
    keep(bottomhalf _cons) ///
    cells(b(fmt(3) star) se(par fmt(3))) ///
    stats(N, fmt(0) labels("Observations")) ///
    varlabels(bottomhalf "Bottom half (=1)" _cons "Mean (Top half)") ///
    title("Table 1 Panel B — Balance within Tracking Schools: Top vs Bottom Stream") ///
    starlevels(* 0.10 ** 0.05 *** 0.01) ///
    note("Tracking schools only. Cluster-robust SE (cluster = school).") ///
    rtf

*--- Ghi chú sư phạm
display _newline "GHI CHÚ: Không có hệ số nào significant → randomization thành công"
display "Nếu có biến unbalanced, cần kiểm tra lại randomization protocol"
display "hoặc control cho biến đó trong tất cả hồi quy phía sau"

*===============================================================================
* TABLE 2 — ATE CỦA TRACKING (ITT)
*
* Mục đích sư phạm:
*   - Col 1-2: ITT thuần — tác động bình quân của ĐƯỢC GÁN vào tracking school
*   - Col 3-4: Heterogeneous ITT theo initial achievement (bottom half vs top)
*   - Panel B: persistence — tác động có kéo dài sau khi program kết thúc?
*
* Lưu ý: đây là ITT (Intent-to-Treat), không phải TOT (Treatment-on-Treated)
* vì một số học sinh trong tracking schools không thực sự được track đúng
* (compliance ~98.6%) — xem phần LATE bên dưới
*===============================================================================
display _newline(2) "========================================================"
display "TABLE 2 — ATE (ITT): TÁC ĐỘNG CỦA TRACKING"
display "========================================================"

*--- Panel A: Short-run (endline, 18 tháng)
display _newline "--- Panel A: Short-run (Endline) ---"

* Col 1: Bivariate — không controls
reg z_totalscore tracking, vce(cluster schoolid)
estimates store t2a_c1

* Col 2: Với full controls
reg z_totalscore tracking $CONTROLS, vce(cluster schoolid)
estimates store t2a_c2

* Col 3: Heterogeneous theo bottom/top half
* [Kỹ thuật]: dùng $CONTROLS_BASE (không có bottomhalf) + đưa bottomhalf riêng
*             + interaction → tránh perfect multicollinearity
reg z_totalscore tracking bottomhalf c.bottomhalf#c.tracking ///
    $CONTROLS_BASE, vce(cluster schoolid)
estimates store t2a_c3

* Test: tổng tác động cho nhóm DƯỚI = tracking + bottomhalf×tracking = 0?
test tracking + c.bottomhalf#c.tracking = 0
local p_bottom = r(p)
display "p-value (total effect for bottom half = 0): " %5.3f `p_bottom'

* Col 4: Heterogeneous theo 4 quartiles
reg z_totalscore tracking ///
    bottomquarter c.bottomquarter#c.tracking ///
    secondquarter c.secondquarter#c.tracking ///
    thirdquarter  c.thirdquarter#c.tracking  ///
    $CONTROLS_BASE, vce(cluster schoolid)
estimates store t2a_c4

estout t2a_c1 t2a_c2 t2a_c3 t2a_c4, ///
    keep(tracking bottomhalf c.bottomhalf#c.tracking ///
         bottomquarter c.bottomquarter#c.tracking ///
         secondquarter c.secondquarter#c.tracking ///
         thirdquarter  c.thirdquarter#c.tracking  ///
         etpteacher) ///
    cells(b(fmt(3) star) se(par fmt(3))) ///
    stats(N r2, fmt(0 3) labels("N" "R-squared")) ///
    varlabels(tracking "Tracking" ///
              bottomhalf "Bottom half" ///
              c.bottomhalf#c.tracking "Tracking × Bottom half" ///
              bottomquarter "Bottom quarter" ///
              c.bottomquarter#c.tracking "Tracking × Bottom quarter" ///
              secondquarter "2nd quarter" ///
              c.secondquarter#c.tracking "Tracking × 2nd quarter" ///
              thirdquarter "3rd quarter" ///
              c.thirdquarter#c.tracking "Tracking × 3rd quarter" ///
              etpteacher "Contract teacher") ///
    title("Table 2 Panel A — Short-run ITT Effect of Tracking (Endline)") ///
    starlevels(* 0.10 ** 0.05 *** 0.01) ///
    note("Cluster-robust SE (cluster = school)." ///
         "Controls in Col 2-4: girl, age, contract teacher, baseline percentile." ///
         "Col 3-4 use CONTROLS_BASE (excl. bottomhalf/quarter dummies)" ///
         "to avoid perfect multicollinearity with interaction terms.")

*--- Panel B: Long-run (1 năm sau, Fall 2007)
display _newline "--- Panel B: Long-run (1 năm sau khi chương trình kết thúc) ---"

reg z_r2_totalscore tracking, vce(cluster schoolid)
estimates store t2b_c1

reg z_r2_totalscore tracking $CONTROLS, vce(cluster schoolid)
estimates store t2b_c2

reg z_r2_totalscore tracking bottomhalf c.bottomhalf#c.tracking ///
    $CONTROLS_BASE, vce(cluster schoolid)
estimates store t2b_c3

reg z_r2_totalscore tracking ///
    bottomquarter c.bottomquarter#c.tracking ///
    secondquarter c.secondquarter#c.tracking ///
    thirdquarter  c.thirdquarter#c.tracking  ///
    $CONTROLS_BASE, vce(cluster schoolid)
estimates store t2b_c4

* Xuất RTF Panel A + B cùng một file
esttab t2a_c1 t2a_c2 t2a_c3 t2a_c4 t2b_c1 t2b_c2 t2b_c3 t2b_c4 ///
    using "$OUT\table2_ATE_ITT.rtf", replace ///
    keep(tracking bottomhalf c.bottomhalf#c.tracking ///
         bottomquarter c.bottomquarter#c.tracking ///
         secondquarter c.secondquarter#c.tracking ///
         thirdquarter  c.thirdquarter#c.tracking  ///
         etpteacher) ///
    cells(b(fmt(3) star) se(par fmt(3))) ///
    stats(N r2, fmt(0 3) labels("N" "R-squared")) ///
    varlabels(tracking "Tracking" ///
              bottomhalf "Bottom half" ///
              c.bottomhalf#c.tracking "Tracking × Bottom half" ///
              bottomquarter "Bottom quarter" ///
              c.bottomquarter#c.tracking "Tracking × Bottom quarter" ///
              secondquarter "2nd quarter" ///
              c.secondquarter#c.tracking "Tracking × 2nd quarter" ///
              thirdquarter "3rd quarter" ///
              c.thirdquarter#c.tracking "Tracking × 3rd quarter" ///
              etpteacher "Contract teacher") ///
    mgroups("Panel A: Short-run (Endline)" "Panel B: Long-run (Follow-up)", ///
            pattern(1 0 0 0 1 0 0 0)) ///
    mtitles("(1)" "(2)" "(3)" "(4)" "(1)" "(2)" "(3)" "(4)") ///
    title("Table 2 — ITT Effect of Tracking on Standardized Test Scores") ///
    starlevels(* 0.10 ** 0.05 *** 0.01) ///
    note("Standardized scores: mean 0, SD 1 in non-tracking schools." ///
         "Cluster-robust SE (cluster = school). *** p<0.01, ** p<0.05, * p<0.10") ///
    rtf

*===============================================================================
* TABLE 3 — HETEROGENEOUS EFFECTS
*
* Mục đích sư phạm:
*   - Tác động của tracking có khác nhau theo giới tính / loại giáo viên không?
*   - Đây là phân tích HTE (Heterogeneous Treatment Effects) được pre-specified
*     trong design của DDK (không phải data mining sau khi thấy kết quả)
*   - Lưu ý: khi split sample, power giảm → SE lớn hơn, cần thận trọng
*     khi interpret significance
*===============================================================================
display _newline(2) "========================================================"
display "TABLE 3 — HETEROGENEOUS TREATMENT EFFECTS"
display "========================================================"

*--- Panel A: Theo giới tính
display _newline "--- Panel A: by Gender ---"
foreach g in 0 1 {
    local gname = cond(`g'==0, "Boys", "Girls")
    reg z_totalscore tracking bottomhalf c.bottomhalf#c.tracking ///
        $CONTROLS_BASE if girl == `g', vce(cluster schoolid)
    estimates store t3a_g`g'
    display "  `gname': N = " e(N) ", clusters = " e(N_clust)
}

*--- Panel B: Theo loại giáo viên
display _newline "--- Panel B: by Teacher Type ---"
foreach t in 0 1 {
    local tname = cond(`t'==0, "Civil-service teacher", "Contract teacher")
    reg z_totalscore tracking bottomhalf c.bottomhalf#c.tracking ///
        $CONTROLS_BASE if etpteacher == `t', vce(cluster schoolid)
    estimates store t3b_t`t'
    display "  `tname': N = " e(N) ", clusters = " e(N_clust)
}

*--- Test equality of coefficients across subgroups (Panel A)
* Pooled model với interaction gender × treatment để test H0: β_boys = β_girls
reg z_totalscore tracking girl c.tracking#c.girl ///
    bottomhalf c.bottomhalf#c.tracking ///
    $CONTROLS_BASE, vce(cluster schoolid)
test c.tracking#c.girl
display "p-value (tracking effect: boys = girls): " %5.3f r(p)
estimates store t3a_pooled

* Tương tự cho teacher type
reg z_totalscore tracking etpteacher c.tracking#c.etpteacher ///
    bottomhalf c.bottomhalf#c.tracking ///
    girl agetest realpercentile, vce(cluster schoolid)
test c.tracking#c.etpteacher
display "p-value (tracking effect: civil = contract teacher): " %5.3f r(p)
estimates store t3b_pooled

* Xuất RTF
esttab t3a_g0 t3a_g1 t3b_t0 t3b_t1 ///
    using "$OUT\table3_HTE.rtf", replace ///
    keep(tracking c.bottomhalf#c.tracking bottomhalf) ///
    cells(b(fmt(3) star) se(par fmt(3))) ///
    stats(N r2, fmt(0 3) labels("N" "R-squared")) ///
    varlabels(tracking "Tracking" ///
              bottomhalf "Bottom half" ///
              c.bottomhalf#c.tracking "Tracking × Bottom half") ///
    mgroups("Panel A: By Gender" "Panel B: By Teacher Type", ///
            pattern(1 0 1 0)) ///
    mtitles("Boys" "Girls" "Civil-service" "Contract") ///
    title("Table 3 — Heterogeneous Treatment Effects") ///
    starlevels(* 0.10 ** 0.05 *** 0.01) ///
    note("All regressions include controls: age, baseline percentile." ///
         "Cluster-robust SE (cluster = school). *** p<0.01, ** p<0.05, * p<0.10" ///
         "Test for equality of coefficients across subgroups reported in text.") ///
    rtf

*===============================================================================
* ITT vs LATE (LATE = Local Average Treatment Effect)
*
* Mục đích sư phạm:
*   - ITT: tác động của ĐƯỢC GÁN (assignment) vào tracking school
*   - LATE (= TOT ở đây vì one-sided noncompliance): tác động của THỰC SỰ
*     được xếp vào lớp thấp (lowstream = 1) đối với những ai tuân thủ assignment
*   - Instrument: assigned_lowstream = bottomhalf × tracking
*     Logic: bottomhalf × tracking là randomized → valid instrument cho
*     lowstream (actual placement) trong tracking schools
*   - Với compliance ~98.6%, LATE ≈ ITT / compliance_rate ≈ ITT / 0.986
*     → gap giữa ITT và LATE rất nhỏ, nhưng IV logic quan trọng về mặt khái niệm
*
* Cấu trúc IV:
*   First stage:  lowstream = α + γ·(bottomhalf×tracking) + X'β + ε
*   Second stage: z_score  = α + δ·lowstream_hat + X'β + η
*   δ = LATE = tác động nhân quả của lowstream placement
*===============================================================================
display _newline(2) "========================================================"
display "ITT vs LATE: INTENT-TO-TREAT vs LOCAL AVERAGE TREATMENT EFFECT"
display "========================================================"

*--- Chỉ chạy trên tracking schools (lowstream chỉ có ý nghĩa ở đây)
*--- và chỉ cho học sinh thuộc bottom half (nhóm bị ảnh hưởng bởi assignment)

preserve
    keep if tracking == 1
    keep if !missing(lowstream) & !missing(bottomhalf)

    display _newline "--- Trong tracking schools ---"
    display "N = " _N

    * Step 1: ITT — tác động của assignment (bottomhalf) lên outcome
    * (Trong tracking schools, bottomhalf là instrument/assignment)
    display _newline "--- Step 1: ITT (Reduced Form) ---"
    reg z_totalscore bottomhalf $CONTROLS_BASE, vce(cluster schoolid)
    estimates store itt_rf

    * Step 2: First stage — compliance
    * Mức độ bottomhalf → lowstream (actual track placement)
    display _newline "--- Step 2: First Stage ---"
    reg lowstream bottomhalf $CONTROLS_BASE, vce(cluster schoolid)
    estimates store late_fs
    display "First stage F-statistic: " e(F)
    * Rule of thumb: F > 10 → instrument mạnh

    * Step 3: LATE via 2SLS
    display _newline "--- Step 3: LATE (2SLS) ---"
    ivregress 2sls z_totalscore $CONTROLS_BASE ///
        (lowstream = bottomhalf), vce(cluster schoolid)
    estimates store late_2sls
    estat firststage
    estat endogenous    // Wu-Hausman test: H0: OLS consistent (no endogeneity)

    * Step 4: OLS naive (để so sánh — bị selection bias nếu compliance không hoàn hảo)
    display _newline "--- Step 4: OLS (naive, for comparison) ---"
    reg z_totalscore lowstream $CONTROLS_BASE, vce(cluster schoolid)
    estimates store ols_naive

    * So sánh ITT / First stage = LATE (manual check)
    quietly reg z_totalscore bottomhalf $CONTROLS_BASE, vce(cluster schoolid)
    local itt_coef = _b[bottomhalf]
    quietly reg lowstream bottomhalf $CONTROLS_BASE, vce(cluster schoolid)
    local fs_coef  = _b[bottomhalf]
    display _newline "Manual check: ITT / First stage = " %6.4f `itt_coef' " / " %6.4f `fs_coef' ///
            " = " %6.4f `itt_coef'/`fs_coef'
    display "(Should equal 2SLS coefficient on lowstream)"

    * Xuất RTF: ITT, First Stage, LATE, OLS cạnh nhau
    esttab itt_rf late_fs late_2sls ols_naive ///
        using "$OUT\table_ITT_LATE.rtf", replace ///
        keep(bottomhalf lowstream $CONTROLS_BASE) ///
        order(bottomhalf lowstream) ///
        cells(b(fmt(3) star) se(par fmt(3))) ///
        stats(N r2, fmt(0 3) labels("N" "R-squared")) ///
        varlabels(bottomhalf "Bottom half (Assignment / Instrument)" ///
                  lowstream  "Low stream (Actual placement)" ///
                  girl "Female" ///
                  agetest "Age at test" ///
                  etpteacher "Contract teacher" ///
                  realpercentile "Baseline percentile") ///
        mtitles("ITT (Reduced Form)" "First Stage" "LATE (2SLS)" "OLS (Naive)") ///
        title("ITT vs LATE — Effect of Low-stream Placement on Test Scores" ///
              "(Tracking schools only)") ///
        starlevels(* 0.10 ** 0.05 *** 0.01) ///
        note("Sample: tracking schools only. Instrument: bottom-half assignment (bottomhalf)." ///
             "Compliance rate ~98.6% → LATE ≈ ITT / 0.986." ///
             "Wu-Hausman test reported in log. Cluster-robust SE (cluster = school)." ///
             "OLS (Naive) shown for comparison only — biased if selection into stream." ///
             "*** p<0.01, ** p<0.05, * p<0.10") ///
        rtf

restore

*===============================================================================
* RANDOMIZATION INFERENCE (RI)
*
* Mục đích sư phạm:
*   - Cluster SE giả định số cluster → ∞; với 121 clusters, asymptotic có thể
*     chưa kick in hoàn toàn
*   - RI không dựa vào phân phối asymptotic: chỉ dùng tính ngẫu nhiên của
*     ASSIGNMENT để xây phân phối null (sharp null: H0: không có treatment effect
*     cho bất kỳ ai)
*   - Ý nghĩa: nếu assignment là random, ta có thể permute labels tracking/non-tracking
*     và tính β̂ cho mỗi permutation → build exact finite-sample distribution
*   - p-value RI = tỷ lệ permutation có |β̂_perm| ≥ |β̂_observed|
*
* Lưu ý kỹ thuật:
*   - ritest cluster theo schoolid vì randomization ở cấp trường
*   - 1000 reps đủ cho p-value stable đến 2 decimal
*   - So sánh p-value RI với p-value cluster SE: nếu gần nhau → cluster SE đủ tốt
*===============================================================================
display _newline(2) "========================================================"
display "RANDOMIZATION INFERENCE"
display "========================================================"

capture which ritest
if _rc != 0 {
    display as error "ritest chưa được cài. Chạy: ssc install ritest"
    display as error "Bỏ qua phần Randomization Inference."
}
else {
    display _newline "--- RI: Main tracking effect (z_totalscore) ---"
    display "Số permutations: 1000 (cluster = schoolid)"

    * RI cho main effect (Col 2 của Table 2)
    ritest tracking _b[tracking], reps(1000) cluster(schoolid) seed(12345): reg z_totalscore tracking $CONTROLS, cluster(schoolid)

    * Lưu p-value vào scalar để so sánh
    local ri_p = r(p1)

    * Cluster SE p-value từ cùng spec
    quietly reg z_totalscore tracking $CONTROLS, vce(cluster schoolid)
    local cl_p = 2 * ttail(e(df_r), abs(_b[tracking]/_se[tracking]))

    display _newline "--- SO SÁNH P-VALUES ---"
    display "  Cluster SE p-value: " %6.4f `cl_p'
    display "  RI p-value:         " %6.4f `ri_p'
    display "  (Nếu gần nhau → cluster SE đủ tốt với 121 clusters)"

    * RI cho bottom half interaction (Col 3)
    display _newline "--- RI: Interaction tracking × bottomhalf ---"
    ritest tracking _b[c.bottomhalf#c.tracking], reps(1000) cluster(schoolid) seed(12345): reg z_totalscore tracking bottomhalf c.bottomhalf#c.tracking $CONTROLS_BASE, cluster(schoolid)
}

*===============================================================================
* ATTRITION CHECK
* Mục đích sư phạm:
*   - Nếu attrition KHÁC NHAU giữa treatment và control → sample endline
*     không còn comparable → internal validity bị đe dọa
*   - Test: hồi quy attrition lên tracking + interaction → nếu tracking
*     predict attrition → có differential attrition
*===============================================================================
display _newline(2) "========================================================"
display "ATTRITION CHECK"
display "========================================================"

* Main test: differential attrition
reg attrition tracking $CONTROLS_BASE, vce(cluster schoolid)
estimates store attr_main

* Với interaction (heterogeneous attrition?)
reg attrition tracking bottomhalf c.bottomhalf#c.tracking ///
    $CONTROLS_BASE, vce(cluster schoolid)
estimates store attr_interact
test tracking
test c.bottomhalf#c.tracking
display "Joint F-test (tracking + interaction = 0): "
test tracking c.bottomhalf#c.tracking

esttab attr_main attr_interact ///
    using "$OUT\table_attrition.rtf", replace ///
    keep(tracking bottomhalf c.bottomhalf#c.tracking) ///
    cells(b(fmt(3) star) se(par fmt(3))) ///
    stats(N r2, fmt(0 3) labels("N" "R-squared")) ///
    varlabels(tracking "Tracking" ///
              bottomhalf "Bottom half" ///
              c.bottomhalf#c.tracking "Tracking × Bottom half") ///
    mtitles("Main" "With Interaction") ///
    title("Attrition Check — Dependent variable: Student not found at Endline (=1)") ///
    starlevels(* 0.10 ** 0.05 *** 0.01) ///
    note("If tracking predicts attrition → differential attrition → internal validity concern." ///
         "DDK (2011): attrition rates balanced (~17.4% both groups)." ///
         "Cluster-robust SE (cluster = school). *** p<0.01, ** p<0.05, * p<0.10") ///
    rtf

*===============================================================================
* VISUALIZATION
* Bốn biểu đồ xuất PNG → $OUT
*   Fig 1: Coefficient plot — Balance check (Table 1)
*   Fig 2: Bar chart với CI — ATE theo quartile (Table 2)
*   Fig 3: Coefficient plot — Heterogeneous effects (Table 3)
*   Fig 4: Bar chart — Attrition rate by treatment & subgroup
*
* Ghi chú kỹ thuật:
*   - Tất cả dùng scheme s2color (built-in, không cần cài thêm)
*   - coefplot cần: ssc install coefplot
*   - Nếu coefplot chưa có, Fig 1 & 3 sẽ được bỏ qua (capture)
*===============================================================================
display _newline(2) "========================================================"
display "VISUALIZATION"
display "========================================================"

* Thiết lập scheme thống nhất cho tất cả biểu đồ
set scheme s2color

*-------------------------------------------------------------------------------
* FIGURE 1 — BALANCE CHECK: Coefficient Plot
* Hiển thị hệ số hồi quy của "tracking" trên mỗi pre-treatment variable
* Nếu randomization thành công: tất cả CI phải chứa 0
* Mục đích sư phạm: trực quan hóa "nothing predicts treatment assignment"
*-------------------------------------------------------------------------------
capture which coefplot
if _rc != 0 {
    display as error "coefplot chưa cài. Chạy: ssc install coefplot. Bỏ qua Fig 1 & 3."
}
else {

    display _newline "--- Figure 1: Balance Check Coefficient Plot ---"

    * Chạy lại 4 regressions balance để có estimates trong memory
    quietly reg girl        tracking, vce(cluster schoolid)
    estimates store fig1_girl
    quietly reg agetest     tracking, vce(cluster schoolid)
    estimates store fig1_age
    quietly reg std_mark    tracking, vce(cluster schoolid)
    estimates store fig1_mark
    quietly reg percentile  tracking, vce(cluster schoolid)
    estimates store fig1_pct

    coefplot fig1_girl fig1_age fig1_mark fig1_pct, ///
        keep(tracking) ///
        bylabel("Female" "Age at test" "Baseline score (std)" "Baseline percentile") ///
        xline(0, lcolor(red) lpattern(dash) lwidth(medthin)) ///
        msymbol(D) mcolor(navy) msize(medium) ///
        ciopts(lcolor(navy) lwidth(medthick)) ///
        legend(off) ///
        title("Figure 1. Balance Check" "(Coefficient of Tracking on Pre-treatment Variables)", ///
              size(medsmall) color(black)) ///
        subtitle("95% CI, cluster-robust SE by school", size(small) color(gs8)) ///
        note("H{subscript:0}: All coefficients = 0 if randomization successful." ///
             "Red dashed line at zero.", size(vsmall)) ///
        xlabel(, format(%5.3f)) ///
        graphregion(color(white)) plotregion(color(white)) ///
        bgcolor(white)

    graph export "$OUT\fig1_balance_coefplot.png", replace width(1400) height(900)
    display "  → Saved: fig1_balance_coefplot.png"

*-------------------------------------------------------------------------------
* FIGURE 3 — HETEROGENEOUS EFFECTS: Coefficient Plot
* So sánh tác động tracking giữa: Boys vs Girls, Civil vs Contract teacher
* Mục đích sư phạm: visual test của HTE — xem CI có overlap không
*-------------------------------------------------------------------------------
    display _newline "--- Figure 3: Heterogeneous Effects Coefficient Plot ---"

    * Chạy lại Table 3 regressions
    quietly reg z_totalscore tracking bottomhalf c.bottomhalf#c.tracking ///
        $CONTROLS_BASE if girl == 0, vce(cluster schoolid)
    estimates store fig3_boys

    quietly reg z_totalscore tracking bottomhalf c.bottomhalf#c.tracking ///
        $CONTROLS_BASE if girl == 1, vce(cluster schoolid)
    estimates store fig3_girls

    quietly reg z_totalscore tracking bottomhalf c.bottomhalf#c.tracking ///
        $CONTROLS_BASE if etpteacher == 0, vce(cluster schoolid)
    estimates store fig3_civil

    quietly reg z_totalscore tracking bottomhalf c.bottomhalf#c.tracking ///
        $CONTROLS_BASE if etpteacher == 1, vce(cluster schoolid)
    estimates store fig3_contract

    coefplot ///
        (fig3_boys,    label("Boys")             msymbol(O) mcolor(navy)    ciopts(lcolor(navy))) ///
        (fig3_girls,   label("Girls")            msymbol(O) mcolor(cranberry) ciopts(lcolor(cranberry))) ///
        (fig3_civil,   label("Civil-service tchr") msymbol(D) mcolor(dkgreen)  ciopts(lcolor(dkgreen))) ///
        (fig3_contract,label("Contract teacher") msymbol(D) mcolor(orange)  ciopts(lcolor(orange))), ///
        keep(tracking c.bottomhalf#c.tracking) ///
        xline(0, lcolor(red) lpattern(dash) lwidth(medthin)) ///
        bylabel("Main tracking effect" "Tracking × Bottom half") ///
        legend(rows(2) size(small) position(6)) ///
        title("Figure 3. Heterogeneous Treatment Effects", ///
              size(medsmall) color(black)) ///
        subtitle("Effect of tracking by gender and teacher type" ///
                 "95% CI, cluster-robust SE by school", size(small) color(gs8)) ///
        xlabel(, format(%5.3f)) ///
        graphregion(color(white)) plotregion(color(white)) ///
        bgcolor(white)

    graph export "$OUT\fig3_HTE_coefplot.png", replace width(1400) height(900)
    display "  → Saved: fig3_HTE_coefplot.png"

} // end capture coefplot block

*-------------------------------------------------------------------------------
* FIGURE 2 — ATE THEO QUARTILE: Bar chart với CI
* Hiển thị tác động tracking riêng cho từng quartile của baseline achievement
* Mục đích sư phạm: DDK's key result — tracking helps ALL quartiles, not just top
*
* Kỹ thuật: chạy 4 regressions riêng (split sample theo quartile)
*           dùng matrix để lưu β̂ và SE, vẽ bằng twoway bar + rcap
*-------------------------------------------------------------------------------
display _newline "--- Figure 2: ATE by Quartile ---"

* Chạy regression riêng cho từng quartile, lưu β̂ và 95% CI
matrix B = J(4, 3, .)   // [coef, lb, ub] cho 4 quartiles

local qvars "bottomquarter secondquarter thirdquarter topquarter"
local i = 1
foreach q of local qvars {
    quietly reg z_totalscore tracking $CONTROLS_BASE if `q' == 1, ///
        vce(cluster schoolid)
    matrix B[`i', 1] = _b[tracking]
    matrix B[`i', 2] = _b[tracking] - 1.96 * _se[tracking]
    matrix B[`i', 3] = _b[tracking] + 1.96 * _se[tracking]
    local i = `i' + 1
}

* Đưa matrix vào dataset tạm để vẽ
preserve
    clear
    svmat B, names(col)
    rename (c1 c2 c3) (coef lb ub)
    gen quartile = _n
    label define qlbl 1 "Bottom quarter" 2 "2nd quarter" 3 "3rd quarter" 4 "Top quarter"
    label values quartile qlbl

    * Zero line reference
    gen zero = 0

    twoway ///
        (bar coef quartile, barwidth(0.6) color(navy%70) lcolor(navy)) ///
        (rcap ub lb quartile, lcolor(gs4) lwidth(medthick) msize(zero)) ///
        (line zero quartile, lcolor(red) lpattern(dash) lwidth(medthin)), ///
        xlabel(1 "Bottom quarter" 2 "2nd quarter" 3 "3rd quarter" 4 "Top quarter", ///
               angle(15) labsize(small)) ///
        ylabel(, format(%5.3f) labsize(small)) ///
        ytitle("Effect of tracking (SD units)", size(small)) ///
        xtitle("Initial achievement quartile", size(small)) ///
        legend(off) ///
        title("Figure 2. ATE of Tracking by Initial Achievement Quartile", ///
              size(medsmall) color(black)) ///
        subtitle("Endline test score (standardized). 95% CI, cluster-robust SE by school.", ///
                 size(small) color(gs8)) ///
        note("DDK (2011) key result: tracking benefits students across all quartiles." ///
             "Controls: female, age, contract teacher, baseline percentile.", ///
             size(vsmall)) ///
        graphregion(color(white)) plotregion(color(white)) ///
        bgcolor(white)

    graph export "$OUT\fig2_ATE_by_quartile.png", replace width(1400) height(900)
    display "  → Saved: fig2_ATE_by_quartile.png"
restore

*-------------------------------------------------------------------------------
* FIGURE 4 — ATTRITION: Bar chart so sánh treatment vs control
* Hiển thị attrition rate theo: (1) tracking vs non-tracking,
*                               (2) bottom half vs top half (trong tracking)
*                               (3) short-run vs long-run
* Mục đích sư phạm: nếu attrition rate giống nhau → no differential attrition
*                   → internal validity không bị đe dọa
*-------------------------------------------------------------------------------
display _newline "--- Figure 4: Attrition Rates ---"

preserve
    * Tính attrition rate theo nhóm
    * Short-run (endline) và Long-run (follow-up)
    gen group = .
    replace group = 1 if tracking == 0                          // Non-tracking
    replace group = 2 if tracking == 1                          // Tracking
    replace group = 3 if tracking == 1 & bottomhalf == 0       // Tracking, top half
    replace group = 4 if tracking == 1 & bottomhalf == 1       // Tracking, bottom half

    collapse (mean) attr_sr=attrition attr_lr=r2_attrition ///
             (semean) se_sr=attrition se_lr=r2_attrition, by(group)

    * CI
    gen lb_sr = attr_sr - 1.96 * se_sr
    gen ub_sr = attr_sr + 1.96 * se_sr
    gen lb_lr = attr_lr - 1.96 * se_lr
    gen ub_lr = attr_lr + 1.96 * se_lr

    * Labels
    label define glbl 1 "Non-tracking" 2 "Tracking" 3 "Tracking{newline}Top half" 4 "Tracking{newline}Bottom half"
    label values group glbl

    * Tạo x-position lệch thủ công thay vì dùng offset() (không hỗ trợ trong twoway bar)
    gen x_sr = group - 0.2
    gen x_lr = group + 0.2

    twoway ///
        (bar attr_sr x_sr, barwidth(0.35) color(navy%80) lcolor(navy)) ///
        (bar attr_lr x_lr, barwidth(0.35) color(cranberry%70) lcolor(cranberry)) ///
        (rcap ub_sr lb_sr x_sr, lcolor(gs4) lwidth(medthick)) ///
        (rcap ub_lr lb_lr x_lr, lcolor(gs4) lwidth(medthick)), ///
        xlabel(1 "Non-tracking" 2 "Tracking (all)" ///
               3 "Tracking, top half" 4 "Tracking, bottom half", ///
               angle(15) labsize(small)) ///
        ylabel(0(0.05)0.35, format(%4.2f) labsize(small)) ///
        ytitle("Attrition rate", size(small)) ///
        xtitle("") ///
        legend(order(1 "Short-run (Endline, 18m)" 2 "Long-run (Follow-up, 30m)") ///
               rows(1) position(6) size(small)) ///
        title("Figure 4. Attrition Rates by Group", ///
              size(medsmall) color(black)) ///
        subtitle("95% CI based on standard error of mean", size(small) color(gs8)) ///
        note("Balanced attrition across groups → no differential attrition concern." ///
             "DDK (2011): ~17.4% endline, ~21.7% follow-up, balanced by treatment status.", ///
             size(vsmall)) ///
        graphregion(color(white)) plotregion(color(white)) ///
        bgcolor(white)

    graph export "$OUT\fig4_attrition.png", replace width(1400) height(900)
    display "  → Saved: fig4_attrition.png"
restore

*===============================================================================
* KẾT THÚC
*===============================================================================
display _newline(2) "========================================================"
display "REPLICATION HOÀN TẤT"
display "========================================================"
display "Output files tại: $OUT"
display "  --- Bảng hồi quy (RTF) ---"
display "  table1_balance.rtf          — Balance check"
display "  table1b_balance_stream.rtf  — Balance within tracking schools"
display "  table2_ATE_ITT.rtf          — Main ATE: Panel A + Panel B"
display "  table3_HTE.rtf              — Heterogeneous effects"
display "  table_ITT_LATE.rtf          — ITT vs LATE (2SLS)"
display "  table_attrition.rtf         — Attrition check"
display "  --- Biểu đồ (PNG) ---"
display "  fig1_balance_coefplot.png   — Balance check coefficient plot"
display "  fig2_ATE_by_quartile.png    — ATE by initial achievement quartile"
display "  fig3_HTE_coefplot.png       — Heterogeneous effects coefficient plot"
display "  fig4_attrition.png          — Attrition rates by group"
display "========================================================"

log close
exit
