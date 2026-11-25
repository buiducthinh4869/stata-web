/*==========================================================================
  THUC HANH STATA — BUOI 7: PROPENSITY SCORE MATCHING (PSM)
  Du lieu: Cattaneo (2010) — Hut thuoc khi mang thai va can nang so sinh
  Lenh nap du lieu: webuse cattaneo2, clear

  Hoc phan : Danh gia tac dong chinh sach dinh luong
  Giang vien: TS. Do Ngoc Kien — Truong Dai hoc Ngoai thuong

  Cau truc do-file:
    Buoc 0 : Thiet lap moi truong
    Buoc 1 : Nap va mo ta du lieu
    Buoc 2 : Uoc luong Propensity Score (Logit)
    Buoc 3 : Kiem tra vung ho tro chung (Common Support)
    Buoc 4 : Thuc hien Matching (Nearest Neighbor)
    Buoc 5 : Kiem tra can bang sau matching (Balance Check)
    Buoc 6 : Uoc luong ATT tren mau da match
    Buoc 7 : Sensitivity analysis (thay doi so nguoi duoc ghep)

  Ghi chu  : Du lieu co san trong Stata, khong can tai tu ben ngoai.
             Thu muc "ketqua" se duoc tao tu dong.
             Yeu cau: package psmatch2 (cai qua ssc)
==========================================================================*/


/*--------------------------------------------------------------------------
  BUOC 0: THIET LAP MOI TRUONG
--------------------------------------------------------------------------*/

clear all
set more off
set linesize 120

* Tao thu muc ket qua
capture mkdir "ketqua"

* Cai package can thiet
capture which psmatch2
if _rc ssc install psmatch2, replace

capture which outreg2
if _rc ssc install outreg2, replace

capture which coefplot
if _rc ssc install coefplot, replace

di as text "=== Thiet lap hoan tat ==="


/*--------------------------------------------------------------------------
  BUOC 1: NAP VA MO TA DU LIEU

  Boi canh nghien cuu:
    Cau hoi: Viec me hut thuoc khi mang thai co lam giam can nang
             cua tre so sinh khong?

  Van de nhan qua:
    Khong the so sanh truc tiep can nang giua con cua me hut thuoc
    va me khong hut thuoc vi hai nhom nay KHAC NHAU ve nhieu dac
    diem (tuoi, hoc van, tinh trang hon nhan, dinh duong...).
    → Can phai "can bang" hai nhom truoc khi so sanh.

  Phuong phap PSM (nhu cong cu lam sach mau):
    Buoc 1: Uoc luong xac suat hut thuoc (propensity score) dua
            tren cac dac diem quan sat duoc.
    Buoc 2: Ghep moi ba me hut thuoc voi ba me khong hut thuoc
            co propensity score gan nhat.
    Buoc 3: Uoc luong ATT tren mau da can bang.

  Luu y:
    PSM chi kiem soat SELECTION ON OBSERVABLES —
    khong xu ly duoc unobserved confounders.
    Do do PSM la buoc lam sach mau, khong phai nguon nhan qua.
--------------------------------------------------------------------------*/

di as text _newline "=== BUOC 1: NAP DU LIEU ==="

webuse cattaneo2, clear

* Tong quan du lieu
di _newline "--- Tong quan du lieu ---"
describe
di _newline "--- Thong ke mo ta theo nhom ---"
bysort mbsmoke: summarize bweight mage medu mmarried ///
    mhisp alcohol deadkids nprenatal mrace

* Kiem tra ty le hut thuoc
di _newline "--- Ty le hut thuoc trong mau ---"
tab mbsmoke, missing

* So sanh thu (naive): can nang trung binh hai nhom TRUOC matching
di _newline "--- So sanh thu (naive, TRUOC matching) ---"
ttest bweight, by(mbsmoke)

* Luu ATT naive de so sanh sau
qui ttest bweight, by(mbsmoke)
scalar att_naive = r(mu_1) - r(mu_2)
di as text "Chenh lech can nang TRUOC matching (naive): " ///
   as result %6.1f att_naive as text " gram"
di as text "(Day la so lieu TRUOC khi can bang — co the bi confounded)"


/*--------------------------------------------------------------------------
  BUOC 2: UOC LUONG PROPENSITY SCORE

  Propensity Score: e(X) = P(D=1 | X) — xac suat duoc treatment
  (hut thuoc) dieu kien tren tap hop cac bien kiem soat X.

  Uoc luong bang logit: bien phu thuoc la mbsmoke (0/1),
  bien doc lap la cac dac diem quan sat duoc cua ba me.

  Covariates chon:
    mage      — tuoi ba me (anh huong hanh vi hut thuoc)
    medu      — hoc van ba me (anh huong nhan thuc suc khoe)
    mmarried  — tinh trang hon nhan (anh huong ho tro xa hoi)
    mhisp     — nguoc goc goc Tay Ban Nha (kiem soat van hoa)
    alcohol   — uong ruou khi mang thai (hanh vi suc khoe tuong quan)
    deadkids  — so lan mat con (kiem soat lich su sinh san)
    mrace     — chung toc (kiem soat chenh lech xa hoi)
    nprenatal — so lan kham truoc sinh (kiem soat tiep can y te)
    order     — thu tu sinh (anh huong dieu kien gia dinh)
--------------------------------------------------------------------------*/

di as text _newline "=== BUOC 2: UOC LUONG PROPENSITY SCORE ==="

* --- Uoc luong logit ---
logit mbsmoke mage medu mmarried mhisp alcohol deadkids ///
    mrace nprenatal order, nolog

* Luu ket qua logit
estimates store logit_ps
outreg2 using "ketqua/bang_logit_ps.rtf", replace ///
    title("Bang ket qua Logit — Mo hinh Propensity Score") ///
    ctitle("P(Hut thuoc = 1 | X)") ///
    addnote("Bien phu thuoc: mbsmoke (1 = me hut thuoc khi mang thai)." ///
            "Phuong phap: Logit MLE. SE trong ngoac don." ///
            "* p<0.10  ** p<0.05  *** p<0.01") ///
    label dec(3) word
di as text "-> Da luu: ketqua/bang_logit_ps.rtf"

* Tinh va luu propensity score
predict ps_score, pr
label var ps_score "Propensity Score — P(hut thuoc | X)"
summarize ps_score, detail

* --- Bieu do phan phoi propensity score truoc matching ---
twoway ///
    (kdensity ps_score if mbsmoke == 0, ///
        lcolor(navy) lwidth(medthick) lpattern(solid)) ///
    (kdensity ps_score if mbsmoke == 1, ///
        lcolor(cranberry) lwidth(medthick) lpattern(dash)) ///
    , ///
    xlabel(0(0.1)1, labsize(small)) ///
    ylabel(, labsize(small) angle(0)) ///
    xtitle("Propensity Score — P(hut thuoc | X)", size(medsmall)) ///
    ytitle("Mat do", size(medsmall)) ///
    title("Phan phoi Propensity Score — TRUOC Matching", ///
          size(medlarge) color(black)) ///
    subtitle("Hai phan phoi cang gan nhau → hai nhom cang tuong dong", ///
             size(small) color(gs6)) ///
    legend(order(1 "Khong hut thuoc (control)" ///
                 2 "Hut thuoc (treated)") ///
           size(small) position(1) ring(0) ///
           region(lcolor(gs12) fcolor(white)) cols(1)) ///
    note("Vung chong lap cua hai phan phoi = vung ho tro chung (common support)." ///
         "Chi co the ghep cap duoc trong vung nay.", ///
         size(vsmall) color(gs6)) ///
    scheme(s2color) graphregion(color(white)) plotregion(color(white))

graph export "ketqua/bieu_do_ps_truoc_matching.png", replace width(1400)
di as text "-> Da luu: ketqua/bieu_do_ps_truoc_matching.png"


/*--------------------------------------------------------------------------
  BUOC 3: KIEM TRA VUNG HO TRO CHUNG (COMMON SUPPORT)

  Common Support: vung trong do ca treated va control deu co quan sat.
  Chi co the uoc luong ATT trong vung nay vi ngoai vung nay khong co
  nhom doi chung phu hop.

  Kiem tra:
    - Do thi overlap (hai phan phoi chong len nhau)
    - Loai bo cac quan sat nam ngoai common support
--------------------------------------------------------------------------*/

di as text _newline "=== BUOC 3: KIEM TRA COMMON SUPPORT ==="

* Tinh min/max PS theo nhom de xac dinh common support
qui sum ps_score if mbsmoke == 0
scalar ps_min_ctrl = r(min)
scalar ps_max_ctrl = r(max)
qui sum ps_score if mbsmoke == 1
scalar ps_min_trt  = r(min)
scalar ps_max_trt  = r(max)

scalar cs_lower = max(ps_min_ctrl, ps_min_trt)
scalar cs_upper = min(ps_max_ctrl, ps_max_trt)

di as text "Common support: [" as result %5.3f cs_lower ///
   as text " , " as result %5.3f cs_upper as text "]"
count if ps_score < cs_lower | ps_score > cs_upper
di as text "So quan sat nam NGOAI common support: " as result `r(N)'

* Bieu do overlap (box plot PS theo nhom)
graph box ps_score, over(mbsmoke, ///
    relabel(1 "Khong hut thuoc" 2 "Hut thuoc")) ///
    title("Overlap: Phan phoi Propensity Score theo Nhom", ///
          size(medlarge) color(black)) ///
    subtitle("Hai hop phai chong len nhau de dam bao common support", ///
             size(small) color(gs6)) ///
    ytitle("Propensity Score", size(medsmall)) ///
    note("Vung chong lap cua hai hop = vung hop le de ghep cap." ///
         "Quan sat ngoai common support se bi loai truoc khi uoc luong.", ///
         size(vsmall) color(gs6)) ///
    scheme(s2color) graphregion(color(white))

graph export "ketqua/bieu_do_common_support.png", replace width(1400)
di as text "-> Da luu: ketqua/bieu_do_common_support.png"


/*--------------------------------------------------------------------------
  BUOC 4: THUC HIEN MATCHING

  Phuong phap: Nearest Neighbor Matching (1-to-1, with replacement)
    - Moi ba me hut thuoc duoc ghep voi 1 ba me khong hut thuoc
      co propensity score gan nhat
    - With replacement: cung 1 ba me control co the duoc chon nhieu lan
      (giu duoc nhieu treated units hon, giam bias)
    - Caliper = 0.05: chi ghep neu PS sai khac < 0.05 sd cua PS
      (tranh ghep cap qua xa nhau)

  Lenh: psmatch2 (Leuven & Sianesi 2003)
--------------------------------------------------------------------------*/

di as text _newline "=== BUOC 4: THUC HIEN MATCHING ==="

* --- Matching chinh: 1-to-1, caliper = 0.05 sd(PS) ---
qui sum ps_score
local caliper_val = 0.05 * r(sd)
di as text "Caliper (0.05 x SD propensity score): " as result %7.5f `caliper_val'

psmatch2 mbsmoke, ///
    pscore(ps_score) ///
    outcome(bweight) ///
    neighbor(1) ///
    caliper(`caliper_val') ///
    common ///
    noreplace

* Luu ATT tu psmatch2
scalar att_psm1 = r(att)
scalar se_psm1  = r(seatt)
di as text _newline "--- Ket qua Matching (1-to-1, caliper) ---"
di as text "ATT (PSM 1:1)  = " as result %7.3f att_psm1 as text " gram"
di as text "SE             = " as result %7.3f se_psm1
di as text "t-stat         = " as result %7.3f att_psm1/se_psm1

* Thong ke mau sau matching
di _newline "--- Thong ke mau sau matching ---"
count if _weight > 0 & _weight != .
di as text "So treated duoc match thanh cong: " as result `r(N)'


/*--------------------------------------------------------------------------
  BUOC 5: KIEM TRA CAN BANG SAU MATCHING (BALANCE CHECK)

  Muc tieu quan trong nhat cua PSM:
  Sau khi match, hai nhom phai CO BANG NHAU tren tat ca covariates —
  tuc la standardized mean difference (SMD) < 0.1 cho moi bien.

  SMD = (mean_treated - mean_control) / sd_pooled_before_matching

  Neu SMD sau matching > 0.1: matching that bai, nen thu:
    - Them/bot covariates trong logit
    - Doi caliper
    - Dung phuong phap matching khac (kernel, radius)

  Hai bieu do:
    5a. Love plot (covariate balance truoc vs sau matching)
    5b. Bieu do phan phoi PS sau matching
--------------------------------------------------------------------------*/

di as text _newline "=== BUOC 5: KIEM TRA CAN BANG SAU MATCHING ==="

* Tinh SMD bang tay cho tung covariate
local covs "mage medu mmarried mhisp alcohol deadkids mrace nprenatal order"
local n_covs : word count `covs'

tempfile smd_data
postfile smd_post str15 varname smd_before smd_after ///
    using `smd_data', replace

foreach v of local covs {
    * SMD truoc matching
    qui sum `v' if mbsmoke == 1
    local m1 = r(mean)
    local s1 = r(sd)
    qui sum `v' if mbsmoke == 0
    local m0 = r(mean)
    local s0 = r(sd)
    local sd_pooled = sqrt((`s1'^2 + `s0'^2) / 2)
    local smd_b = abs(`m1' - `m0') / `sd_pooled'

    * SMD sau matching (dung weight _weight)
    qui sum `v' if mbsmoke == 1 & _weight != .
    local m1_m = r(mean)
    qui sum `v' [iweight = _weight] if mbsmoke == 0 & _weight != .
    local m0_m = r(mean)
    local smd_a = abs(`m1_m' - `m0_m') / `sd_pooled'

    post smd_post ("`v'") (`smd_b') (`smd_a')

    di as text "  `v':"
    if `smd_a' < 0.1 {
        di as text "    SMD truoc = " as result %6.4f `smd_b' ///
           as text "   SMD sau = " as result %6.4f `smd_a' ///
           as text "   OK"
    }
    else {
        di as text "    SMD truoc = " as result %6.4f `smd_b' ///
           as text "   SMD sau = " as result %6.4f `smd_a' ///
           as error "   CANH BAO > 0.1"
    }
}
postclose smd_post

* --- Love Plot: SMD truoc vs sau matching ---
preserve
    use `smd_data', clear
    gen varnum = _n
    labmask varnum, values(varname)
    local nv = _N

    twoway ///
        (scatter varnum smd_before, ///
            msymbol(circle) mcolor(cranberry) msize(large)) ///
        (scatter varnum smd_after, ///
            msymbol(diamond) mcolor(navy) msize(large)) ///
        , ///
        xline(0.1, lcolor(orange) lpattern(dash) lwidth(medium)) ///
        xline(0, lcolor(gs12) lpattern(solid) lwidth(thin)) ///
        ylabel(1(1)`nv', valuelabel labsize(medsmall) angle(0)) ///
        xlabel(0(0.1)0.6, labsize(small)) ///
        xtitle("Standardized Mean Difference (SMD)", size(medsmall)) ///
        ytitle("") ///
        title("Love Plot — Kiem tra Can bang Covariates", ///
              size(medlarge) color(black)) ///
        subtitle("SMD < 0.1 (duong cam) = can bang tot sau matching", ///
                 size(small) color(gs6)) ///
        legend(order(1 "Truoc matching" 2 "Sau matching (PSM 1:1)") ///
               size(small) position(4) ring(0) ///
               region(lcolor(gs12) fcolor(white)) cols(1)) ///
        note("Duong cam doc (SMD = 0.1): nguong chap nhan trong tai lieu." ///
             "Tat ca cac diem sau matching nen nam ben trai duong nay.", ///
             size(vsmall) color(gs6)) ///
        scheme(s2color) ///
        graphregion(color(white) margin(l=15)) ///
        plotregion(color(white) margin(medsmall))

    graph export "ketqua/bieu_do_love_plot.png", replace width(1400)
    di as text "-> Da luu: ketqua/bieu_do_love_plot.png"
restore

* --- Phan phoi PS sau matching ---
twoway ///
    (kdensity ps_score if mbsmoke == 0 & _weight != ., ///
        lcolor(navy) lwidth(medthick) lpattern(solid)) ///
    (kdensity ps_score if mbsmoke == 1 & _weight != ., ///
        lcolor(cranberry) lwidth(medthick) lpattern(dash)) ///
    , ///
    xlabel(0(0.1)1, labsize(small)) ///
    ylabel(, labsize(small) angle(0)) ///
    xtitle("Propensity Score", size(medsmall)) ///
    ytitle("Mat do", size(medsmall)) ///
    title("Phan phoi Propensity Score — SAU Matching", ///
          size(medlarge) color(black)) ///
    subtitle("Hai phan phoi nen chong len nhau hoan toan sau matching", ///
             size(small) color(gs6)) ///
    legend(order(1 "Khong hut thuoc (control)" ///
                 2 "Hut thuoc (treated)") ///
           size(small) position(1) ring(0) ///
           region(lcolor(gs12) fcolor(white)) cols(1)) ///
    note("So sanh voi bieu do TRUOC matching:" ///
         "Neu hai phan phoi gan nhau hon → matching thanh cong.", ///
         size(vsmall) color(gs6)) ///
    scheme(s2color) graphregion(color(white)) plotregion(color(white))

graph export "ketqua/bieu_do_ps_sau_matching.png", replace width(1400)
di as text "-> Da luu: ketqua/bieu_do_ps_sau_matching.png"


/*--------------------------------------------------------------------------
  BUOC 6: UOC LUONG ATT TREN MAU DA MATCH

  Sau khi co mau can bang, uoc luong tac dong bang hoi quy OLS
  tren mau da match — voi bien treatment va covariates.

  Hai buoc:
    6a. So sanh don gian (difference in means) tren mau da match
    6b. Regression adjustment: kem them covariates vao OLS
        de giam variance va xu ly bat can bang nho con lai
        sau matching (doubly robust approach)
--------------------------------------------------------------------------*/

di as text _newline "=== BUOC 6: UOC LUONG ATT ==="

* Tao mau matched
* _weight > 0 la cac quan sat duoc giu lai
gen matched_sample = (_weight > 0 & _weight != .) | ///
                     (mbsmoke == 1 & _weight != .)
replace matched_sample = 0 if matched_sample == .

* --- 6a. Difference in means tren mau da match ---
di as text _newline "--- 6a. Difference in means (mau da match) ---"
ttest bweight if matched_sample == 1, by(mbsmoke)

scalar att_ttest = r(mu_1) - r(mu_2)
scalar se_ttest  = r(se)
di as text "ATT (t-test sau matching): " as result %7.1f att_ttest as text " gram"

* --- 6b. OLS tren mau da match (regression adjustment) ---
di as text _newline "--- 6b. OLS voi covariates (mau da match) ---"

* Mo hinh 1: Don gian (khong covariate)
reg bweight mbsmoke if matched_sample == 1, vce(robust)
estimates store M_simple
outreg2 using "ketqua/bang_ket_qua_psm.rtf", replace ///
    title("Bang ket qua uoc luong ATT — PSM Cattaneo (2010)") ///
    ctitle("(1) Don gian") ///
    addnote("Bien phu thuoc: bweight (can nang so sinh, gram)." ///
            "Mau: chi gom cac quan sat da duoc ghep cap trong buoc PSM." ///
            "SE robust trong ngoac don. * p<0.10  ** p<0.05  *** p<0.01") ///
    label dec(1) word

* Mo hinh 2: Them covariates (doubly robust)
reg bweight mbsmoke mage medu mmarried mhisp alcohol ///
    deadkids mrace nprenatal order if matched_sample == 1, vce(robust)
estimates store M_covs
outreg2 using "ketqua/bang_ket_qua_psm.rtf", append ///
    ctitle("(2) + Covariates") ///
    label dec(1) word

* Mo hinh 3: OLS tren mau CHUA match (de so sanh)
reg bweight mbsmoke mage medu mmarried mhisp alcohol ///
    deadkids mrace nprenatal order, vce(robust)
estimates store M_full
outreg2 using "ketqua/bang_ket_qua_psm.rtf", append ///
    ctitle("(3) OLS toan bo mau") ///
    addnote("Mo hinh (3) dung toan bo mau chua match — de so sanh voi (1) va (2).") ///
    label dec(1) word

di as text "-> Da luu: ketqua/bang_ket_qua_psm.rtf"

* Luu he so tu cac mo hinh
scalar att_m1 = _b[mbsmoke]
qui reg bweight mbsmoke mage medu mmarried mhisp alcohol ///
    deadkids mrace nprenatal order if matched_sample == 1, vce(robust)
scalar att_m2 = _b[mbsmoke]
qui reg bweight mbsmoke mage medu mmarried mhisp alcohol ///
    deadkids mrace nprenatal order, vce(robust)
scalar att_m3 = _b[mbsmoke]


/*--------------------------------------------------------------------------
  BUOC 7: SENSITIVITY ANALYSIS — THAY DOI SO NGUOI DUOC GHEP

  PSM 1:1 co the bo qua nhieu thong tin. Kiem tra xem ket qua co
  thay doi khong khi dung 2 hoac 3 nguoi kem cap (k-nearest neighbor).

  Dong thoi ve bieu do tom tat so sanh ATT qua cac phuong phap:
    - Naive (chua match)
    - PSM 1:1
    - PSM 1:2
    - PSM 1:3
    - OLS (toan bo mau, co covariates)
--------------------------------------------------------------------------*/

di as text _newline "=== BUOC 7: SENSITIVITY ANALYSIS ==="

* PSM 1:2 (with replacement — bat buoc khi neighbor > 1)
psmatch2 mbsmoke, pscore(ps_score) outcome(bweight) ///
    neighbor(2) caliper(`caliper_val') common
scalar att_k2 = r(att)
scalar se_k2  = r(seatt)
di as text "ATT (PSM 1:2): " as result %7.1f att_k2 as text " gram"

* PSM 1:3 (with replacement)
psmatch2 mbsmoke, pscore(ps_score) outcome(bweight) ///
    neighbor(3) caliper(`caliper_val') common
scalar att_k3 = r(att)
scalar se_k3  = r(seatt)
di as text "ATT (PSM 1:3): " as result %7.1f att_k3 as text " gram"

* Khoi phuc matching chinh (1:1) de _weight dung
psmatch2 mbsmoke, pscore(ps_score) outcome(bweight) ///
    neighbor(1) caliper(`caliper_val') common noreplace

* --- Lay SE naive ---
qui ttest bweight, by(mbsmoke)
scalar se_naive = r(se)

* --- Lay SE OLS full sample ---
qui reg bweight mbsmoke mage medu mmarried mhisp alcohol ///
    deadkids mrace nprenatal order, vce(robust)
scalar att_ols  = _b[mbsmoke]
scalar se_ols   = _se[mbsmoke]

* --- Tao dataset so sanh ATT qua cac phuong phap ---
preserve
    clear
    set obs 5

    gen str30 method  = ""
    gen att_est = .
    gen ci_lo   = .
    gen ci_hi   = .

    replace method  = "(1) Naive (chua match)"  in 1
    replace att_est = att_naive                  in 1
    replace ci_lo   = att_naive - 1.96*se_naive  in 1
    replace ci_hi   = att_naive + 1.96*se_naive  in 1

    replace method  = "(2) PSM 1:1"             in 2
    replace att_est = att_psm1                   in 2
    replace ci_lo   = att_psm1 - 1.96*se_psm1   in 2
    replace ci_hi   = att_psm1 + 1.96*se_psm1   in 2

    replace method  = "(3) PSM 1:2"             in 3
    replace att_est = att_k2                     in 3
    replace ci_lo   = att_k2 - 1.96*se_k2       in 3
    replace ci_hi   = att_k2 + 1.96*se_k2       in 3

    replace method  = "(4) PSM 1:3"             in 4
    replace att_est = att_k3                     in 4
    replace ci_lo   = att_k3 - 1.96*se_k3       in 4
    replace ci_hi   = att_k3 + 1.96*se_k3       in 4

    replace method  = "(5) OLS (toan bo mau)"   in 5
    replace att_est = att_ols                    in 5
    replace ci_lo   = att_ols - 1.96*se_ols      in 5
    replace ci_hi   = att_ols + 1.96*se_ols      in 5

    gen methodnum = _n
    labmask methodnum, values(method)

    twoway ///
        (rcap ci_lo ci_hi methodnum, horizontal ///
            lcolor(navy) lwidth(medthick)) ///
        (scatter methodnum att_est, ///
            mcolor(navy) msymbol(circle) msize(large)) ///
        , ///
        xline(0, lcolor(cranberry) lpattern(dash) lwidth(medium)) ///
        ylabel(1(1)5, valuelabel labsize(medsmall) angle(0)) ///
        xlabel(, labsize(small) angle(0)) ///
        xtitle("ATT uoc luong (gram)", size(medsmall)) ///
        ytitle("") ///
        title("So sanh ATT qua cac phuong phap", ///
              size(medlarge) color(black)) ///
        subtitle("Can nang so sinh thap hon cua tre co me hut thuoc (gram)", ///
                 size(small) color(gs6)) ///
        note("ATT: Average Treatment Effect on the Treated." ///
             "CI 95%. Tat ca uoc luong nen nhat quan ve dau (am = giam can nang)." ///
             "Muc do tin cay phu thuoc vao gia dinh selection on observables.", ///
             size(vsmall) color(gs6)) ///
        scheme(s2color) ///
        graphregion(color(white) margin(l=25)) ///
        plotregion(color(white) margin(medsmall))

    graph export "ketqua/bieu_do_so_sanh_phuong_phap.png", replace width(1400)
    di as text "-> Da luu: ketqua/bieu_do_so_sanh_phuong_phap.png"
restore


/*--------------------------------------------------------------------------
  TONG KET
--------------------------------------------------------------------------*/

di as text _newline "=== HOAN THANH ==="
di as text "{hline 65}"
di as text "  KET QUA DA LUU: thu muc ketqua/"
di as text "{hline 65}"
di as text "  BANG (RTF — mo bang Microsoft Word):"
di as text "    bang_logit_ps.rtf      — Mo hinh logit Propensity Score"
di as text "    bang_ket_qua_psm.rtf   — Ket qua uoc luong ATT"
di as text _newline "  BIEU DO (PNG):"
di as text "    bieu_do_ps_truoc_matching.png  — PS truoc matching"
di as text "    bieu_do_common_support.png     — Common support"
di as text "    bieu_do_love_plot.png          — Balance check (Love plot)"
di as text "    bieu_do_ps_sau_matching.png    — PS sau matching"
di as text "    bieu_do_so_sanh_phuong_phap.png — So sanh ATT"
di as text "{hline 65}"
di as text _newline "  TOM TAT KET QUA:"
di as text "  ATT Naive (chua match)  = " as result %7.1f att_naive as text " gram"
di as text "  ATT PSM 1:1             = " as result %7.1f att_psm1  as text " gram"
di as text "  ATT PSM 1:2             = " as result %7.1f att_k2    as text " gram"
di as text "  ATT PSM 1:3             = " as result %7.1f att_k3    as text " gram"
di as text "  ATT OLS (toan bo mau)   = " as result %7.1f att_m3    as text " gram"
di as text "{hline 65}"
di as text "  KET LUAN: Hut thuoc khi mang thai lam giam can nang"
di as text "  so sinh khoang " as result %4.0f abs(att_psm1) as text "–" ///
   as result %4.0f abs(att_k3) as text " gram (ATT, PSM 1:1 den 1:3)."
di as text "  (Ket qua chi co gia tri neu gia dinh selection on"
di as text "   observables duoc dap ung — xem ghi chu phuong phap)"
di as text "{hline 65}"

/*==========================================================================
  GHI CHU PHUONG PHAP CHO NGHIEN CUU SINH

  1. PSM LY TUONG DE DAY CAI GI?
     PSM minh hoa ro nhat van de selection bias va cach lam sach mau.
     Nhung quan trong hon: PSM giup NCS hieu tai sao OLS thong thuong
     co the bi confounded khi hai nhom qua khac nhau (model dependence).

  2. HAN CHE COT LÕI CUA PSM:
     PSM chi xu ly SELECTION ON OBSERVABLES — neu co bien khong quan sat
     duoc anh huong ca viec hut thuoc lan can nang (vi du: stress, che do
     dinh duong khong do duoc), ATT van bi bias. PSM khong phai nguon
     nhan qua manh nhu DiD hay RDD.

  3. TAI SAO DUNG PSM + REGRESSION (DOUBLY ROBUST)?
     Ket hop PSM (de can bang mau) voi OLS co covariates (de xu ly
     bat can bang nho con lai) cho ket qua tot hon chi dung mot trong hai.
     Day la "doubly robust": nhat quan neu IT NHAT MOT trong hai buoc dung.

  4. LOVE PLOT LA TIEU CHI QUAN TRONG NHAT:
     SMD < 0.1 sau matching moi la thanh cong. Bao cao ATT ma khong
     co Love plot la thieu thong tin quan trong nhat cua PSM.

  5. KING & NIELSEN (2019):
     CEM (Coarsened Exact Matching) thuong cho balance tot hon PSM
     vi khong collapse thong tin da chieu cua X ve mot scalar.
     PSM van day vi direct quan hon — nhung NCS can biet co lua chon tot hon.
==========================================================================*/
