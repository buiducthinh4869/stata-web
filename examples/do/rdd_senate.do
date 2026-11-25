/*==========================================================================
  THUC HANH STATA — BUOI 6: REGRESSION DISCONTINUITY DESIGN (RDD)
  Du lieu: Cattaneo, Frandsen & Titiunik (2015) — U.S. Senate Elections
  
  Hoc phan : Danh gia tac dong chinh sach dinh luong
  Giang vien: TS. Do Ngoc Kien — Truong Dai hoc Ngoai thuong

  Cau truc do-file:
    Buoc 0 : Thiet lap moi truong
    Buoc 1 : Nap va kiem tra du lieu
    Buoc 2 : RD Plot — truc quan hoa discontinuity
    Buoc 3 : Uoc luong RDD bang rdrobust
    Buoc 4 : McCrary density test — kiem tra manipulation
    Buoc 5 : Covariate balance / Placebo test

  Ghi chu  : Dat do-file nay cung thu muc voi rdrobust_senate.dta.
             Thu muc "ketqua" se duoc tao tu dong.
==========================================================================*/


/*--------------------------------------------------------------------------
  BUOC 0: THIET LAP MOI TRUONG
--------------------------------------------------------------------------*/

clear all
set more off
set linesize 120

* Tao thu muc ket qua (neu chua ton tai)
capture mkdir "ketqua"

* Cai package can thiet (chi chay lan dau)
capture which rdrobust
if _rc {
    net install rdrobust, ///
        from("https://raw.githubusercontent.com/rdpackages/rdrobust/master/stata") ///
        replace
}

capture which rddensity
if _rc {
    net install rddensity, ///
        from("https://raw.githubusercontent.com/rdpackages/rddensity/master/stata") ///
        replace
}

capture which lpdensity
if _rc {
    net install lpdensity, ///
        from("https://raw.githubusercontent.com/nppackages/lpdensity/master/stata") ///
        replace
}

capture which outreg2
if _rc ssc install outreg2, replace

capture which coefplot
if _rc ssc install coefplot, replace

di as text "=== Thiet lap hoan tat ==="
/*
findit rdrobust_kweight
findit labmak
Cài package mới nhất cho 2 thư mục trên
*/

/*--------------------------------------------------------------------------
  BUOC 1: NAP VA KIEM TRA DU LIEU
--------------------------------------------------------------------------*/

di as text _newline "=== BUOC 1: NAP DU LIEU ==="

use "rdrobust_senate.dta", clear

describe
summarize

/*
  CAU TRUC DU LIEU:
  - 1,390 quan sat: cac cuoc bau cu Thuong vien My
  - margin  : running variable — chenh lech ty le phieu Đan chu so voi
              ung cu vien manh nhat cua doi lap. Cutoff = 0:
              margin > 0 → Đan chu thang (treated)
              margin < 0 → Đan chu thua (control)
  - vote    : outcome — ty le phieu bau cua Đan chu ky tiep theo
  - termssenate, termshouse: kinh nghiem chinh tri (covariate)
  - demwinprv1, demwinprv2 : thang ky truoc 1, 2 (placebo outcomes)
  - presdemvotesh~1        : ty le phieu tong thong Đan chu (covariate)
*/

* Kiem tra phan phoi running variable
di _newline "--- Phan bo quan sat quanh nguong ---"
count if margin >= -10 & margin <= 10
count if margin >= 0
count if margin < 0

* Tao bien treatment
gen treated = (margin >= 0)
label var treated "Đan chu thang (margin >= 0)"
tab treated


/*--------------------------------------------------------------------------
  BUOC 2: RD PLOT — TRUC QUAN HOA DISCONTINUITY

  Muc tieu: Nhin thay bang mat discontinuity tai nguong margin = 0.
  Dung rdplot (tu package rdrobust) — binscatter voi CI bands.

  Hai bieu do:
    2a. RD plot: vote (t+1) theo margin (t)
    2b. Phan phoi running variable — phat hien clustering/manipulation
--------------------------------------------------------------------------*/

di as text _newline "=== BUOC 2: RD PLOT ==="

* --- 2a. RD Plot chinh: outcome (vote) theo running variable (margin) ---
rdplot vote margin, ///
    c(0) ///
    nbins(20 20) ///
    binselect(es) ///
    kernel(triangular) ///
    title("RD Plot: Ty le phieu bau ky tiep theo") ///
    y_label("Ty le phieu Đan chu ky t+1 (%)") ///
    x_label("Chenh lech phieu Đan chu ky t (margin)") ///
    graph_options( ///
        xline(0, lcolor(cranberry) lpattern(dash) lwidth(medium)) ///
        note("Nguong cutoff: margin = 0 (Đan chu thang neu margin > 0)." ///
             "Cac diem la gia tri trung binh trong tung bin. Nguon: Cattaneo et al. (2015).", ///
             size(vsmall) color(gs6)) ///
        graphregion(color(white)) plotregion(color(white)) ///
        scheme(s2color) ///
    )

graph export "ketqua/bieu_do_rd_plot.png", replace width(1400)
di as text "-> Da luu: ketqua/bieu_do_rd_plot.png"

* --- 2b. Histogram phan phoi running variable ---
twoway ///
    (histogram margin if margin < 0, ///
        width(2) color(cranberry%50) lcolor(cranberry)) ///
    (histogram margin if margin >= 0, ///
        width(2) color(navy%50) lcolor(navy)) ///
    , ///
    xline(0, lcolor(black) lpattern(solid) lwidth(medium)) ///
    xlabel(-100(20)100, labsize(small)) ///
    ylabel(, labsize(small) angle(0)) ///
    xtitle("Chenh lech phieu Đan chu (margin)", size(medsmall)) ///
    ytitle("So quan sat", size(medsmall)) ///
    title("Phan phoi Running Variable", size(medlarge) color(black)) ///
    subtitle("Kiem tra so bo ve manipulation tai nguong cutoff = 0", ///
             size(small) color(gs6)) ///
    legend(order(1 "Đan chu thua (control)" 2 "Đan chu thang (treated)") ///
           size(small) position(6) rows(1) region(lcolor(none))) ///
    note("Neu co manipulation: phan phoi se khong lien tuc tai nguong (dot pha hoac but gap)." ///
         "Kiem tra chinh thuc bang McCrary density test o Buoc 4.", ///
         size(vsmall) color(gs6)) ///
    scheme(s2color) graphregion(color(white)) plotregion(color(white))

graph export "ketqua/bieu_do_phan_phoi_margin.png", replace width(1400)
di as text "-> Da luu: ketqua/bieu_do_phan_phoi_margin.png"


/*--------------------------------------------------------------------------
  BUOC 3: UOC LUONG RDD BANG RDROBUST

  Mo hinh RDD co ban:
    E[Y | X] co discontinuity tai X = c (margin = 0)
    RD estimate = lim_{x->0+} E[Y|X=x] - lim_{x->0-} E[Y|X=x]

  rdrobust su dung:
    - Local linear regression (bac 1)
    - Triangular kernel
    - MSE-optimal bandwidth (CCT 2014)
    - Bias-corrected robust confidence intervals (Calonico, Cattaneo & Titiunik 2014)

  Ba mo hinh:
    M1: Baseline (bandwidth tu dong CCT)
    M2: Them covariates (termssenate, presdemvotesh~1)
    M3: Bandwidth thu cong (bandwidth x 0.5 va x 1.5 — kiem tra robustness)
--------------------------------------------------------------------------*/

di as text _newline "=== BUOC 3: UOC LUONG RDD ==="

* --- Mo hinh 1: RDD co ban ---
di _newline "--- Mo hinh 1: RDD co ban (bandwidth CCT) ---"
rdrobust vote margin, c(0) kernel(triangular) bwselect(mserd)
estimates store M1_basic

* Luu ket qua M1 vao scalars de xuat bang sau
scalar h_M1    = e(h_l)
scalar tau_M1  = e(tau_bc)
scalar se_M1   = e(se_tau_rb)
scalar pv_M1   = e(pv_rb)
scalar N_l_M1  = e(N_h_l)
scalar N_r_M1  = e(N_h_r)

di as text "Bandwidth toi uu (CCT): " as result %6.3f h_M1
di as text "RD estimate (bias-corrected): " as result %6.3f tau_M1
di as text "SE (robust): " as result %6.3f se_M1
di as text "p-value: " as result %6.4f pv_M1

* --- Mo hinh 2: RDD voi covariates ---
di _newline "--- Mo hinh 2: RDD + covariates ---"

* Kiem tra ten bien day du
* presdemvoteshlag1 co the bi cat thanh presdemvotesh~1
capture confirm variable presdemvoteshlag1
if _rc {
    * Thu ten bien bi cat
    local covlist "termssenate presdemvotesh~1"
}
else {
    local covlist "termssenate presdemvoteshlag1"
}

rdrobust vote margin, c(0) kernel(triangular) bwselect(mserd) covs(`covlist')
estimates store M2_covs

scalar tau_M2 = e(tau_bc)
scalar se_M2  = e(se_tau_rb)
scalar pv_M2  = e(pv_rb)

di as text "RD estimate voi covariates: " as result %6.3f tau_M2

* --- Mo hinh 3: Sensitivity theo bandwidth ---
di _newline "--- Mo hinh 3: Bandwidth sensitivity ---"

* Bandwidth bang 0.5x
local h_half = h_M1 * 0.5
rdrobust vote margin, c(0) kernel(triangular) h(`h_half')
scalar tau_h05 = e(tau_bc)
scalar se_h05  = e(se_tau_rb)
scalar pv_h05  = e(pv_rb)

* Bandwidth bang 1.5x
local h_15 = h_M1 * 1.5
rdrobust vote margin, c(0) kernel(triangular) h(`h_15')
scalar tau_h15 = e(tau_bc)
scalar se_h15  = e(se_tau_rb)
scalar pv_h15  = e(pv_rb)

di as text "RD estimate (h x 0.5): " as result %6.3f tau_h05
di as text "RD estimate (h x 1.5): " as result %6.3f tau_h15

* --- Xuat bang ket qua chinh bang tay (rdrobust khong ho tro outreg2) ---
di as text _newline "--- Xuat bang ket qua RDD ---"

* Dung putdocx de xuat bang RTF-style, hoac dung file write
* Vi rdrobust khong tuong thich voi outreg2, ta dung estimates table + file write

capture file close fout
file open fout using "ketqua/bang_rdd_ket_qua.txt", write replace

file write fout "Bang ket qua uoc luong RDD — Cuoc bau cu Thuong vien My" _n
file write fout "Bien phu thuoc: Ty le phieu Đan chu ky tiep theo (vote)" _n
file write fout "Running variable: Chenh lech phieu Đan chu (margin), cutoff = 0" _n
file write fout _n
file write fout "----------------------------------------------------------------" _n
file write fout "                          (1)         (2)         (3a)        (3b)" _n
file write fout "                       Co ban    + Covariates  h x 0.5    h x 1.5" _n
file write fout "----------------------------------------------------------------" _n

* RD Estimate
file write fout "RD Estimate        " ///
    %10.3f (tau_M1)  "  " ///
    %10.3f (tau_M2)  "  " ///
    %10.3f (tau_h05) "  " ///
    %10.3f (tau_h15) _n

* SE
file write fout "(SE robust)        " ///
    "(" %8.3f (se_M1)  ")  " ///
    "(" %8.3f (se_M2)  ")  " ///
    "(" %8.3f (se_h05) ")  " ///
    "(" %8.3f (se_h15) ")" _n

* p-value
file write fout "p-value            " ///
    %10.4f (pv_M1)  "  " ///
    %10.4f (pv_M2)  "  " ///
    %10.4f (pv_h05) "  " ///
    %10.4f (pv_h15) _n

file write fout "----------------------------------------------------------------" _n
file write fout "Bandwidth          " %10.3f (h_M1) "  " %10.3f (h_M1) ///
    "  " %10.3f (h_M1*0.5) "  " %10.3f (h_M1*1.5) _n
file write fout "N (trai)           " %10.0f (N_l_M1) _n
file write fout "N (phai)           " %10.0f (N_r_M1) _n
file write fout "----------------------------------------------------------------" _n
file write fout "Kernel: Triangular. Bandwidth: MSE-optimal (CCT 2014)." _n
file write fout "Sai so chuan Bias-corrected Robust trong ngoac don." _n
file write fout "Covariates mo hinh (2): termssenate, presdemvoteshlag1." _n

file close fout
di as text "-> Da luu: ketqua/bang_rdd_ket_qua.txt"

* --- Bieu do sensitivity bandwidth ---
* Tap hop cac RD estimate theo nhieu bandwidth khac nhau
di _newline "--- Ve bieu do sensitivity bandwidth ---"

local bw_list = ""
local tau_list = ""
local ci_lo_list = ""
local ci_hi_list = ""

* Chay rdrobust o 7 muc bandwidth tu 0.4h den 2h
tempfile bw_results
postfile bw_post bw_mult tau_est ci_lo ci_hi using `bw_results', replace

foreach mult in 0.4 0.6 0.8 1.0 1.2 1.5 2.0 {
    local h_cur = h_M1 * `mult'
    quietly rdrobust vote margin, c(0) kernel(triangular) h(`h_cur')
    local tau_cur = e(tau_bc)
    local se_cur  = e(se_tau_rb)
    post bw_post (`mult') (`tau_cur') (`tau_cur' - 1.96*`se_cur') (`tau_cur' + 1.96*`se_cur')
}
postclose bw_post

preserve
    use `bw_results', clear

    twoway ///
        (rcap ci_lo ci_hi bw_mult, lcolor(navy) lwidth(medium)) ///
        (scatter tau_est bw_mult, mcolor(navy) msymbol(circle) msize(large)) ///
        , ///
        yline(0, lcolor(cranberry) lpattern(dash) lwidth(medium)) ///
        xline(1.0, lcolor(gs10) lpattern(dot)) ///
        xlabel(0.4(0.2)2.0, labsize(small)) ///
        ylabel(, labsize(small) angle(0)) ///
        xtitle("Bandwidth (boi so cua bandwidth toi uu CCT)", size(medsmall)) ///
        ytitle("RD Estimate (bias-corrected)", size(medsmall)) ///
        title("Sensitivity cua RD Estimate theo Bandwidth", ///
              size(medlarge) color(black)) ///
        subtitle("Duong cheo: bandwidth toi uu CCT (boi so = 1.0)", ///
                 size(small) color(gs6)) ///
        note("CI 95% bias-corrected robust. Kernel: Triangular." ///
             "Ket qua on dinh khi bandwidth thay doi → ung ho tinh vung cua RDD.", ///
             size(vsmall) color(gs6)) ///
        scheme(s2color) graphregion(color(white)) plotregion(color(white))

    graph export "ketqua/bieu_do_bandwidth_sensitivity.png", replace width(1400)
    di as text "-> Da luu: ketqua/bieu_do_bandwidth_sensitivity.png"
restore


/*--------------------------------------------------------------------------
  BUOC 4: McCRARY DENSITY TEST — KIEM TRA MANIPULATION

  Gia dinh cot loi cua RDD: cac don vi KHONG THE chinh xac kiem soat
  running variable de o dung mot phia cua nguong.

  Neu co manipulation: phan phoi cua running variable se co su but gap
  (discontinuity) tai cutoff — qua nhieu quan sat ngay ben phai nguong
  (nguoi ta "co y" thang sat nut) hoac qua it ben trai.

  Test: rddensity (Cattaneo, Jansson & Ma 2020)
    H0: Phan phoi lien tuc tai cutoff (khong co manipulation)
    Neu p > 0.05: khong bac bo H0 → ung ho tinh hop le cua RDD
--------------------------------------------------------------------------*/

di as text _newline "=== BUOC 4: McCRARY DENSITY TEST ==="

rddensity margin, c(0) plot ///
    graph_opt( ///
        title("McCrary Density Test — Kiem tra Manipulation") ///
        subtitle("Kiem tra tinh lien tuc cua phan phoi tai nguong cutoff = 0") ///
        xtitle("Chenh lech phieu Đan chu (margin)") ///
        ytitle("Mat do uoc luong") ///
        xline(0, lcolor(cranberry) lpattern(dash)) ///
        note("H0: Phan phoi lien tuc tai cutoff (khong co manipulation)." ///
             "Neu p-value > 0.05: khong bac bo H0 → RDD hop le.", ///
             size(vsmall) color(gs6)) ///
        scheme(s2color) ///
        graphregion(color(white)) plotregion(color(white)) ///
    )

graph export "ketqua/bieu_do_density_test.png", replace width(1400)
di as text "-> Da luu: ketqua/bieu_do_density_test.png"

* Luu ket qua density test
scalar pv_density = e(pv_q)
di as text _newline "Ket qua McCrary Density Test:"
di as text "T-statistic: " as result %8.4f e(t_q)
di as text "p-value    : " as result %8.4f e(pv_q)
if e(pv_q) > 0.05 {
    di as text "Ket luan: KHONG bac bo H0 — khong co bang chung manipulation"
    di as text "          → Gia dinh no-manipulation duoc ung ho"
}
else {
    di as text "CANH BAO: Bac bo H0 — co the co manipulation tai nguong"
}


/*--------------------------------------------------------------------------
  BUOC 5: COVARIATE BALANCE VA PLACEBO TEST

  Neu RDD hop le, cac bien predetermine (truoc can thiep) KHONG nen co
  discontinuity tai cutoff — vi thang/thua sat nut la "gan nhu ngau nhien".

  Hai loai kiem tra:
    5a. Covariate balance: chay rdrobust voi covariates lam outcome
        (termssenate, termshouse, presdemvoteshlag1)
        → He so nen khong khac 0

    5b. Placebo cutoff test: chay RDD tai cac nguong gia (fake cutoffs)
        (-20, -10, +10, +20)
        → He so nen khong khac 0 tai cac nguong gia
--------------------------------------------------------------------------*/

di as text _newline "=== BUOC 5: COVARIATE BALANCE VA PLACEBO TEST ==="

* --- 5a. Covariate balance test ---
di _newline "--- 5a. Covariate Balance Test ---"

local covars "termssenate termshouse"

* Kiem tra them presdemvoteshlag1
capture confirm variable presdemvoteshlag1
if _rc == 0 local covars "`covars' presdemvoteshlag1"
capture confirm variable demwinprv1
if _rc == 0 local covars "`covars' demwinprv1"

di as text "Cac bien kiem tra balance: `covars'"

* Xuat ket qua bang coefplot
local i = 0
foreach var of local covars {
    local ++i
    quietly rdrobust `var' margin, c(0) kernel(triangular) bwselect(mserd)
    
    * Luu thanh estimates de coefplot
    * Vi rdrobust khong luu duoi dang standard estimates, dung matrix
    matrix b_cov`i' = e(tau_bc)
    matrix V_cov`i' = e(se_tau_rb)^2
    
    local tau_cov`i' = e(tau_bc)
    local se_cov`i'  = e(se_tau_rb)
    local pv_cov`i'  = e(pv_rb)
    local nm_cov`i'  = "`var'"
    
    di as text "  `var': RD est = " as result %6.3f `tau_cov`i'' ///
       as text ", SE = " as result %6.3f `se_cov`i'' ///
       as text ", p = " as result %6.4f `pv_cov`i''
}
local n_covars = `i'

* Ve bieu do covariate balance bang scatter + rcap thu cong
preserve
    * Tao dataset cho bieu do balance
    clear
    local nrows = `n_covars'
    set obs `nrows'
    gen varname = ""
    gen tau_est = .
    gen ci_lo   = .
    gen ci_hi   = .
    gen varnum  = _n

    forval k = 1/`n_covars' {
        replace varname = "`nm_cov`k''" in `k'
        replace tau_est = `tau_cov`k''  in `k'
        replace ci_lo   = `tau_cov`k'' - 1.96*`se_cov`k'' in `k'
        replace ci_hi   = `tau_cov`k'' + 1.96*`se_cov`k'' in `k'
    }

    * Label truc y
    labmask varnum, values(varname)

    twoway ///
        (rcap ci_lo ci_hi varnum, horizontal lcolor(navy) lwidth(medium)) ///
        (scatter varnum tau_est, mcolor(navy) msymbol(circle) msize(large)) ///
        , ///
        xline(0, lcolor(cranberry) lpattern(dash) lwidth(medium)) ///
        ylabel(1(1)`n_covars', valuelabel labsize(medsmall) angle(0)) ///
        xlabel(, labsize(small) angle(0)) ///
        xtitle("RD Estimate (bias-corrected robust)", size(medsmall)) ///
        ytitle("") ///
        title("Covariate Balance Test", size(medlarge) color(black)) ///
        subtitle("Cac bien predetermine khong nen co discontinuity tai cutoff", ///
                 size(small) color(gs6)) ///
        note("CI 95% bias-corrected robust. Kernel: Triangular, Bandwidth: CCT optimal." ///
             "Neu CI bao gom 0 → khong co discontinuity → RDD hop le.", ///
             size(vsmall) color(gs6)) ///
        scheme(s2color) ///
        graphregion(color(white) margin(l=20)) ///
        plotregion(color(white) margin(medsmall))

    graph export "ketqua/bieu_do_covariate_balance.png", replace width(1400)
    di as text "-> Da luu: ketqua/bieu_do_covariate_balance.png"
restore

* --- 5b. Placebo cutoff test ---
di _newline "--- 5b. Placebo Cutoff Test ---"

local cutoffs "-30 -20 -10 10 20 30"
local n_cut : word count `cutoffs'

tempfile placebo_results
postfile placebo_post cutoff_val tau_est ci_lo ci_hi using `placebo_results', replace

foreach c of local cutoffs {
    * Chi chay khi co du quan sat hai phia
    quietly count if margin >= `c' - 5 & margin < `c'
    local n_left = r(N)
    quietly count if margin >= `c' & margin <= `c' + 5
    local n_right = r(N)
    
    if `n_left' >= 20 & `n_right' >= 20 {
        quietly rdrobust vote margin, c(`c') kernel(triangular) bwselect(mserd)
        local tau_c = e(tau_bc)
        local se_c  = e(se_tau_rb)
        post placebo_post (`c') (`tau_c') (`tau_c' - 1.96*`se_c') (`tau_c' + 1.96*`se_c')
        di as text "  Cutoff = " as result `c' as text ///
           ": RD est = " as result %6.3f `tau_c' ///
           as text ", p = " as result %6.4f e(pv_rb)
    }
    else {
        di as text "  Cutoff = `c': bo qua (khong du quan sat)"
    }
}

* Them cutoff that (c = 0) de so sanh
quietly rdrobust vote margin, c(0) kernel(triangular) bwselect(mserd)
post placebo_post (0) (e(tau_bc)) ///
    (e(tau_bc) - 1.96*e(se_tau_rb)) ///
    (e(tau_bc) + 1.96*e(se_tau_rb))

postclose placebo_post

preserve
    use `placebo_results', clear
    sort cutoff_val

    * Danh dau cutoff that
    gen is_true = (cutoff_val == 0)

    twoway ///
        (rcap ci_lo ci_hi cutoff_val if is_true == 0, ///
            lcolor(gs10) lwidth(medium)) ///
        (scatter tau_est cutoff_val if is_true == 0, ///
            mcolor(gs8) msymbol(circle) msize(large)) ///
        (rcap ci_lo ci_hi cutoff_val if is_true == 1, ///
            lcolor(navy) lwidth(thick)) ///
        (scatter tau_est cutoff_val if is_true == 1, ///
            mcolor(navy) msymbol(circle) msize(vlarge)) ///
        , ///
        yline(0, lcolor(cranberry) lpattern(dash) lwidth(medium)) ///
        xline(0, lcolor(navy) lpattern(dot) lwidth(medium)) ///
        xlabel(`cutoffs' 0, labsize(small) angle(0)) ///
        ylabel(, labsize(small) angle(0)) ///
        xtitle("Gia tri cutoff (that = 0, gia = cac gia tri khac)", size(medsmall)) ///
        ytitle("RD Estimate (bias-corrected robust)", size(medsmall)) ///
        title("Placebo Cutoff Test", size(medlarge) color(black)) ///
        subtitle("RD estimate tai cac nguong gia nen khong khac 0", ///
                 size(small) color(gs6)) ///
        legend(order(2 "Cutoff gia (placebo)" 4 "Cutoff that (c = 0)") ///
               size(small) position(6) rows(1) region(lcolor(none))) ///
        note("CI 95% bias-corrected robust. Kernel: Triangular." ///
             "Neu CI cua cutoff gia bao gom 0 → ung ho tinh hop le cua RDD.", ///
             size(vsmall) color(gs6)) ///
        scheme(s2color) graphregion(color(white)) plotregion(color(white))

    graph export "ketqua/bieu_do_placebo_cutoff.png", replace width(1400)
    di as text "-> Da luu: ketqua/bieu_do_placebo_cutoff.png"
restore


/*--------------------------------------------------------------------------
  TONG KET
--------------------------------------------------------------------------*/

di as text _newline "=== HOAN THANH ==="
di as text "{hline 65}"
di as text "  KET QUA DA LUU VAO THU MUC: ketqua/"
di as text "{hline 65}"
di as text "  BANG SO LIEU (TXT — mo bang Notepad/Word):"
di as text "    bang_rdd_ket_qua.txt      — Ket qua uoc luong RDD"
di as text _newline "  BIEU DO (PNG):"
di as text "    bieu_do_rd_plot.png           — RD Plot chinh"
di as text "    bieu_do_phan_phoi_margin.png  — Phan phoi running variable"
di as text "    bieu_do_bandwidth_sensitivity.png — Sensitivity theo bandwidth"
di as text "    bieu_do_density_test.png      — McCrary density test"
di as text "    bieu_do_covariate_balance.png — Covariate balance test"
di as text "    bieu_do_placebo_cutoff.png    — Placebo cutoff test"
di as text "{hline 65}"
di as text _newline "  TOM TAT KET QUA CHINH:"
di as text "  RD Estimate (co ban) = " as result %6.3f tau_M1 as text " pp"
di as text "  p-value (robust)     = " as result %6.4f pv_M1
di as text "  McCrary p-value      = " as result %6.4f pv_density
di as text "{hline 65}"

/*==========================================================================
  GHI CHU PHUONG PHAP CHO NGHIEN CUU SINH

  1. TAI SAO DUNG LOCAL LINEAR (BAC 1) THAY VI POLYNOMIAL BAC CAO HON?
     Gelman & Imbens (2019, JBES) chi ra rang polynomial bac cao (bac 3, 4)
     rat nhay cam voi cac quan sat o xa nguong — local linear voi bandwidth
     toi uu la lua chon ro rang hon cho RDD.

  2. TAI SAO DUNG BIAS-CORRECTED ROBUST CI?
     Calonico, Cattaneo & Titiunik (2014) chi ra rang CI thong thuong
     cho RDD bi under-coverage. Robust CI tinh den ca uncertainty trong
     uoc luong bias — nen duoc bao cao thay vi CI thong thuong.

  3. TAI SAO KIEM TRA MANIPULATION (McCRARY TEST)?
     Neu cac don vi co the chinh xac dieu chinh running variable de vuot
     nguong (vi du: ky luat chay phieu, tai dem phieu), thi cac don vi ngay
     hai phia cua nguong KHONG "gan nhu ngau nhien" → RDD mat tinh hop le.

  4. LIEN HE VOI LEE (2008):
     Paper goc cua Lee dung data House (2 bien), du lieu Senate nay giau
     bien hon nhung cung cau truc RDD: running variable la margin,
     cutoff = 0, outcome la vote share ky tiep theo.
==========================================================================*/
