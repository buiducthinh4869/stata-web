/*==========================================================================
  THUC HANH STATA — BUOI 7: SYNTHETIC CONTROL METHOD (SCM)
  Du lieu: Abadie, Diamond & Hainmueller (2010, JASA)
           California Tobacco Control Program
  Lenh nap du lieu: use "https://raw.githubusercontent.com/quarcs-lab/data-open/master/synth/synth_smoking.dta", clear  (co san trong package synth2)

  Hoc phan : Danh gia tac dong chinh sach dinh luong
  Giang vien: TS. Do Ngoc Kien — Truong Dai hoc Ngoai thuong

  Cau truc do-file:
    Buoc 0 : Thiet lap moi truong
    Buoc 1 : Nap va mo ta du lieu
    Buoc 2 : Xu huong nen (raw trends)
    Buoc 3 : Uoc luong SCM chinh (synth2 + nested + allopt)
    Buoc 4 : In-space placebo test (built-in synth2)
    Buoc 5 : In-time placebo test (fake treatment year)
    Buoc 6 : Leave-one-out robustness test (loo)
    Buoc 7 : Xuat bang ket qua tong hop

  Ghi chu  : Can cai synth2 truoc:
               ssc install synth, all replace
               net install synth2, from("https://raw.github.com/bquistorff/Stata-modules/master/") replace
             Thu muc "ketqua" se duoc tao tu dong.
             Lenh frame yeu cau Stata 16+.
==========================================================================*/


/*--------------------------------------------------------------------------
  BUOC 0: THIET LAP MOI TRUONG
--------------------------------------------------------------------------*/

clear all
set more off
set linesize 120

* Tao thu muc ket qua
capture mkdir "ketqua"

* Cai package neu chua co
capture which synth2
if _rc {
    ssc install synth, all replace
    net install synth2, ///
        from("https://raw.github.com/bquistorff/Stata-modules/master/") ///
        replace
}

capture which outreg2
if _rc ssc install outreg2, replace

di as text "=== Thiet lap hoan tat ==="


/*--------------------------------------------------------------------------
  BUOC 1: NAP VA MO TA DU LIEU

  Boi canh nghien cuu:
    Thang 11/1988, California thong qua Proposition 99:
      - Tang thue thuoc la them $0.25/bao
      - Cam quang cao thuoc la
      - Tai tro chuong trinh giao duc phong chong hut thuoc

  Cau hoi: Proposition 99 co lam giam tieu thu thuoc la tai CA khong?

  Thiet ke SCM:
    - Treated unit  : California (state = 3)
    - Can thiep     : nam 1989 (Proposition 99 co hieu luc)
    - Donor pool    : 38 bang khac KHONG co chinh sach tuong tu
    - Outcome       : cigsale (bao thuoc la/nam/1000 nguoi)
    - Pre-period    : 1970-1988 (xay dung synthetic CA)
    - Post-period   : 1989-2000 (do tac dong)
    - Predictors    : beer, lnincome, retprice, age15to24,
                      cigsale(1975), cigsale(1980), cigsale(1988)
--------------------------------------------------------------------------*/

di as text _newline "=== BUOC 1: NAP DU LIEU ==="

use "https://raw.githubusercontent.com/quarcs-lab/data-open/master/synth/synth_smoking.dta", clear
tsset state year

describe
di _newline "--- So ban ghi ---"
count
di _newline "--- California (state=3) ---"
list year cigsale if state == 3 & (year == 1970 | year == 1988 | year == 1989 | year == 2000)

* Tinh trung binh dong de ve do thi
bysort year: egen mean_other = mean(cigsale) if state != 3
bysort year: egen mean_donors = max(mean_other)
drop mean_other
label var mean_donors "TB cac bang khac (unweighted)"


/*--------------------------------------------------------------------------
  BUOC 2: XU HUONG NEN — RAW TRENDS

  Truoc khi ap dung SCM, luon ve do thi xu huong thu cong.
  Muc dich:
    - Kiem tra CA co khac biet ro ret voi cac bang khac khong
    - Nhan dien xu huong truoc can thiep (pre-trends)
    - Minh hoa ly do tai sao can SCM thay vi so sanh don gian
--------------------------------------------------------------------------*/

di as text _newline "=== BUOC 2: XU HUONG NEN ==="

* --- 2a. California vs trung binh cac bang ---
twoway ///
    (line cigsale year if state == 3, ///
        lcolor(navy) lwidth(thick) lpattern(solid)) ///
    (line mean_donors year if state == 3, ///
        lcolor(cranberry) lwidth(medium) lpattern(dash)) ///
    , ///
    xline(1989, lcolor(orange) lpattern(dash) lwidth(medium)) ///
    xlabel(1970(5)2000, labsize(small)) ///
    ylabel(40(20)140, labsize(small) angle(0)) ///
    xtitle("Nam", size(medsmall)) ///
    ytitle("Tieu thu thuoc la (bao/nam/1000 nguoi)", size(medsmall)) ///
    title("Xu huong Tieu thu Thuoc la: California vs Dong", ///
          size(medlarge) color(black)) ///
    subtitle("So sanh thu chua kiem soat — can dung SCM de tao counterfactual hop le", ///
             size(small) color(gs6)) ///
    legend(order(1 "California (treated)" ///
                 2 "Trung binh cac bang khac (unweighted)") ///
           size(small) position(1) ring(0) ///
           region(lcolor(gs12) fcolor(white)) cols(1)) ///
    note("Duong doc cam: 1989 (Proposition 99 co hieu luc)." ///
         "So sanh nay CHUA kiem soat cho dac diem khac nhau giua cac bang.", ///
         size(vsmall) color(gs6)) ///
    scheme(s2color) graphregion(color(white)) plotregion(color(white))

graph export "ketqua/bieu_do_xu_huong_nen.png", replace width(1400)
di as text "-> Da luu: ketqua/bieu_do_xu_huong_nen.png"

* --- 2b. Spaghetti plot: tat ca cac bang ---
twoway ///
    (line cigsale year if state != 3, ///
        lcolor(gs13) lwidth(vthin) connect(ascending)) ///
    (line cigsale year if state == 3, ///
        lcolor(navy) lwidth(thick) lpattern(solid) connect(ascending)) ///
    , ///
    xline(1989, lcolor(orange) lpattern(dash) lwidth(medium)) ///
    xlabel(1970(5)2000, labsize(small)) ///
    ylabel(, labsize(small) angle(0)) ///
    xtitle("Nam", size(medsmall)) ///
    ytitle("Tieu thu thuoc la (bao/nam/1000 nguoi)", size(medsmall)) ///
    title("Tieu thu Thuoc la: Tat ca cac Bang (Donor Pool)", ///
          size(medlarge) color(black)) ///
    subtitle("California (xanh dam) trong donor pool (xam nhat)", ///
             size(small) color(gs6)) ///
    legend(order(1 "Donor pool (38 bang)" 2 "California (treated)") ///
           size(small) position(1) ring(0) ///
           region(lcolor(gs12) fcolor(white)) cols(1)) ///
    note("Duong doc cam: 1989. SCM se tim trong so toi uu cua donor pool" ///
         "sao cho to hop tuyen tinh khop voi lich su CA truoc 1989.", ///
         size(vsmall) color(gs6)) ///
    scheme(s2color) graphregion(color(white)) plotregion(color(white))

graph export "ketqua/bieu_do_spaghetti.png", replace width(1400)
di as text "-> Da luu: ketqua/bieu_do_spaghetti.png"


/*--------------------------------------------------------------------------
  BUOC 3: UOC LUONG SCM CHINH

  synth2 options quan trong:
    trunit(3)       : California = don vi xu ly
    trperiod(1989)  : nam can thiep
    xperiod(1980(1)1988) : lay trung binh predictors trong 1980-1988
    nested          : toi uu hoa long (nested optimization) — chinh xac hon
    allopt          : thu 3 diem khoi dau khac nhau — tranh local minima
    frame(california): luu ket qua vao Stata frame "california"
    savegraph(path, replace): luu tat ca do thi synth2 tu dong

  Luu y: allopt chay lau hon nested, nhung cho ket qua chac chan hon.
  Neu muon nhanh hon, bo allopt, chi dung nested.
--------------------------------------------------------------------------*/

di as text _newline "=== BUOC 3: UOC LUONG SCM CHINH ==="

synth2 cigsale ///
    lnincome age15to24 retprice beer ///
    cigsale(1988) cigsale(1980) cigsale(1975) ///
    , ///
    trunit(3) trperiod(1989) ///
    xperiod(1980(1)1988) ///
    nested allopt fig

* Luu ket qua quan trong ngay sau synth2 (truoc khi e() bi ghi de)
scalar rmspe_pre = e(pre_rmspe)
mat W_weights   = e(W_weights)
mat X_balance   = e(X_balance)
mat Y_treated   = e(Y_treated)
mat Y_synthetic = e(Y_synthetic)
mat Y_time      = e(Y_time)

di as text _newline "RMSPE (truoc 1989): " as result %6.3f rmspe_pre
di as text "W weights:"
mat list W_weights

* Luu do thi synth2 vua ve
graph export "ketqua/bieu_do_scm_synth2_auto.png", replace width(1400)
di as text "-> Da luu: ketqua/bieu_do_scm_synth2_auto.png"

* Tao dataset tu e() matrices de ve do thi dep hon
* Y_treated va Y_synthetic la cot vector theo thoi gian
local nobs = rowsof(Y_treated)

preserve
    clear
    set obs `nobs'
    gen year      = .
    gen ca_actual = .
    gen ca_synth  = .
    gen gap       = .

    forval i = 1/`nobs' {
        replace year      = Y_time[`i', 1]    in `i'
        replace ca_actual = Y_treated[`i', 1]  in `i'
        replace ca_synth  = Y_synthetic[`i', 1] in `i'
        replace gap       = ca_actual - ca_synth in `i'
    }

    label var ca_actual "California thuc te"
    label var ca_synth  "Synthetic California"
    label var gap       "Gap = CA thuc - Synthetic CA"

    * Tac dong trung binh theo giai doan
    qui sum gap if year >= 1989
    scalar att_full = r(mean)
    qui sum gap if year >= 1989 & year <= 1994
    scalar att_early = r(mean)
    qui sum gap if year >= 1995
    scalar att_late = r(mean)
    qui sum ca_synth if year >= 1989
    scalar synth_cf = r(mean)
    scalar pct_red = att_full / synth_cf * 100

    di as text _newline "ATT 1989-2000: " as result %6.1f att_full
    di as text "ATT 1989-1994: " as result %6.1f att_early
    di as text "ATT 1995-2000: " as result %6.1f att_late
    di as text "% giam      : " as result %5.1f abs(pct_red) "%"

    * --- Bieu do chinh: CA vs Synthetic CA ---
    twoway ///
        (line ca_actual year, ///
            lcolor(navy) lwidth(thick) lpattern(solid)) ///
        (line ca_synth year, ///
            lcolor(cranberry) lwidth(medium) lpattern(dash)) ///
        , ///
        xline(1989, lcolor(orange) lpattern(dash) lwidth(medium)) ///
        xlabel(1970(5)2000, labsize(small)) ///
        ylabel(40(20)140, labsize(small) angle(0)) ///
        xtitle("Nam", size(medsmall)) ///
        ytitle("Tieu thu thuoc la (bao/nam/1000 nguoi)", size(medsmall)) ///
        title("California thuc te vs Synthetic California", ///
              size(medlarge) color(black)) ///
        subtitle("Tac dong cua Proposition 99 (1989) len tieu thu thuoc la", ///
                 size(small) color(gs6)) ///
        legend(order(1 "California thuc te" ///
                     2 "Synthetic California (counterfactual)") ///
               size(small) position(1) ring(0) ///
               region(lcolor(gs12) fcolor(white)) cols(1)) ///
        note("Duong doc cam: 1989. Khoang cach sau 1989 = tac dong Prop 99." ///
             "RMSPE (1970-1988) = " string(round(rmspe_pre, 0.001)) ///
             " → fit tot truoc can thiep.", ///
             size(vsmall) color(gs6)) ///
        scheme(s2color) graphregion(color(white)) plotregion(color(white))

    graph export "ketqua/bieu_do_scm_chinh.png", replace width(1400)
    di as text "-> Da luu: ketqua/bieu_do_scm_chinh.png"

    * --- Gap plot ---
    twoway ///
        (area gap year if gap < 0 & year >= 1989, ///
            color(cranberry%25) lcolor(cranberry%0)) ///
        (line gap year, ///
            lcolor(navy) lwidth(medthick) lpattern(solid)) ///
        , ///
        xline(1989, lcolor(orange) lpattern(dash) lwidth(medium)) ///
        yline(0, lcolor(gs8) lpattern(solid) lwidth(medium)) ///
        xlabel(1970(5)2000, labsize(small)) ///
        ylabel(, labsize(small) angle(0)) ///
        xtitle("Nam", size(medsmall)) ///
        ytitle("Gap: CA thuc - Synthetic CA", size(medsmall)) ///
        title("Tac dong Chinh sach theo Thoi gian (Gap Plot)", ///
              size(medlarge) color(black)) ///
        subtitle("Gap < 0 sau 1989 = Prop 99 lam GIAM tieu thu thuoc la", ///
                 size(small) color(gs6)) ///
        note("Vung to do: hieu ung chinh sach (CA thuc < Synthetic CA)." ///
             "Gap ≈ 0 truoc 1989 xac nhan synthetic CA khop CA thuc tot.", ///
             size(vsmall) color(gs6)) ///
        scheme(s2color) graphregion(color(white)) plotregion(color(white))

    graph export "ketqua/bieu_do_gap_plot.png", replace width(1400)
    di as text "-> Da luu: ketqua/bieu_do_gap_plot.png"
restore


/*--------------------------------------------------------------------------
  BUOC 4: IN-SPACE PLACEBO TEST (built-in synth2)

  Logic: Neu SCM hop le, gap cua CA sau 1989 nen lon hon gap
  cua cac bang placebo (von khong co Proposition 99).

  placebo(unit cut(2)):
    - Lan luot gan Proposition 99 cho tung bang trong donor pool
    - Chi giu cac bang co pre-RMSPE <= 2x pre-RMSPE cua CA
      (loai cac bang fit kem — khong the so sanh cong bang)
    - Ve tat ca gap len cung do thi

  P-value = ty le bang placebo co gap post >= gap CA post (ve gia tri tuyet doi)
--------------------------------------------------------------------------*/

di as text _newline "=== BUOC 4: IN-SPACE PLACEBO TEST ==="

synth2 cigsale ///
    lnincome age15to24 retprice beer ///
    cigsale(1988) cigsale(1980) cigsale(1975) ///
    , ///
    trunit(3) trperiod(1989) ///
    xperiod(1980(1)1988) ///
    nested ///
    placebo(unit cut(2)) sigf(6) fig

* Luu p-value
scalar pval_inspace = e(p_value)
di as text "P-value (in-space placebo): " as result %5.3f pval_inspace

* Luu do thi hien tai (do thi cuoi cua synth2 la placebo plot)
graph export "ketqua/bieu_do_placebo_inspace.png", replace width(1400)
di as text "-> Da luu: ketqua/bieu_do_placebo_inspace.png"


/*--------------------------------------------------------------------------
  BUOC 5: IN-TIME PLACEBO TEST

  Logic: Neu Proposition 99 thuc su co tac dong, thi gap lon
  chi nen xuat hien vao NAM 1989, khong phai cac nam khac.

  Kiem tra bang cach gia dinh can thiep xay ra vao nam 1985
  (thay vi 1989) va loai bo cigsale(1988) khoi predictors
  (vi 1988 se thuoc giai doan sau can thiep gia).

  Ket qua ky vong:
    - Gap tai "1985 gia" ≈ 0 (khong co sac dong chinh sach thuc su)
    - Neu gap lon → co the co anticipation effect hoac pre-trend
--------------------------------------------------------------------------*/

di as text _newline "=== BUOC 5: IN-TIME PLACEBO TEST ==="

synth2 cigsale ///
    lnincome age15to24 retprice beer ///
    cigsale(1980) cigsale(1975) ///
    , ///
    trunit(3) trperiod(1989) ///
    xperiod(1980(1)1984) ///
    nested ///
    placebo(period(1985)) fig

graph export "ketqua/bieu_do_placebo_intime.png", replace width(1400)
di as text "-> Da luu: ketqua/bieu_do_placebo_intime.png"


/*--------------------------------------------------------------------------
  BUOC 6: LEAVE-ONE-OUT ROBUSTNESS TEST

  Logic: Neu ket qua SCM phu thuoc vao mot bang duy nhat
  trong donor pool (vi du bang do co trong so W rat lon),
  thi loai bang do ra se lam thay doi ket qua dang ke.

  Loo (Leave-One-Out): lan luot loai tung bang co trong so W > 0
  ra khoi donor pool va chay lai SCM.

  Ket qua ky vong:
    - Cac duong loo nen gan voi duong SCM chinh
    - Gap van duy tri ≈ cung muc do — ket qua vung
--------------------------------------------------------------------------*/

di as text _newline "=== BUOC 6: LEAVE-ONE-OUT TEST ==="

synth2 cigsale ///
    lnincome age15to24 retprice beer ///
    cigsale(1988) cigsale(1980) cigsale(1975) ///
    , ///
    trunit(3) trperiod(1989) ///
    xperiod(1980(1)1988) ///
    nested ///
    loo fig

graph export "ketqua/bieu_do_loo.png", replace width(1400)
di as text "-> Da luu: ketqua/bieu_do_loo.png"


/*--------------------------------------------------------------------------
  BUOC 7: XUAT BANG KET QUA TONG HOP
--------------------------------------------------------------------------*/

di as text _newline "=== BUOC 7: XUAT BANG KET QUA TONG HOP ==="

* --- Bang X Balance ---
capture file close fbal
file open fbal using "ketqua/bang_x_balance.txt", write replace

file write fbal "================================================================" _n
file write fbal "BANG KIEM TRA CHAT LUONG FIT — X Balance" _n
file write fbal "California vs Synthetic California vs Trung binh Dong" _n
file write fbal "Predictors: lnincome, age15to24, retprice, beer," _n
file write fbal "            cigsale(1975), cigsale(1980), cigsale(1988)" _n
file write fbal "================================================================" _n
file write fbal _n

local nrows = rowsof(X_balance)
local rnames : rownames X_balance

file write fbal "Predictor              CA thuc  Synthetic CA  TB dong" _n
file write fbal "----------------------------------------------------------------" _n
forval i = 1/`nrows' {
    local vn : word `i' of `rnames'
    file write fbal %22s ("`vn'") " " ///
        %8.2f (X_balance[`i',1]) "   " ///
        %12.2f (X_balance[`i',2]) "   " ///
        %7.2f (X_balance[`i',3]) _n
}
file write fbal "----------------------------------------------------------------" _n
file write fbal "RMSPE (1970-1988) = " %6.3f (rmspe_pre) " bao/nam/1000 nguoi" _n
file write fbal "================================================================" _n
file close fbal
di as text "-> Da luu: ketqua/bang_x_balance.txt"

* --- Bang ket qua tong hop ---
capture file close fout
file open fout using "ketqua/bang_scm_tong_hop.txt", write replace

file write fout "================================================================" _n
file write fout "BANG KET QUA — Synthetic Control Method" _n
file write fout "California Tobacco Control Program (Proposition 99, 1989)" _n
file write fout "Nguon: Abadie, Diamond & Hainmueller (2010, JASA)" _n
file write fout "================================================================" _n
file write fout _n
file write fout "A. CHAT LUONG FIT TRUOC CAN THIEP (1970-1988)" _n
file write fout "----------------------------------------------------------------" _n
file write fout "RMSPE: " %6.3f (rmspe_pre) " bao/nam/1000 nguoi" _n
file write fout "(RMSPE cang nho → synthetic CA cang khop CA thuc)" _n
file write fout _n
file write fout "B. UOC LUONG TAC DONG CHINH SACH" _n
file write fout "----------------------------------------------------------------" _n
file write fout "ATT 1989-1994 (ngan han) : " %7.1f (att_early) " bao/nam/1000 nguoi" _n
file write fout "ATT 1995-2000 (dai han)  : " %7.1f (att_late)  " bao/nam/1000 nguoi" _n
file write fout "ATT 1989-2000 (trung binh): " %6.1f (att_full) " bao/nam/1000 nguoi" _n
file write fout "% giam so voi counterfactual: " %5.1f (abs(pct_red)) "%" _n
file write fout _n
file write fout "C. KIEM TRA TINH HOP LE" _n
file write fout "----------------------------------------------------------------" _n
file write fout "P-value (in-space placebo) : " %5.3f (pval_inspace) _n
file write fout "(Ty le bang placebo co gap >= gap CA theo gia tri tuyet doi)" _n
file write fout _n
file write fout "D. TRONG SO DON VI (W Weights > 0.01)" _n
file write fout "----------------------------------------------------------------" _n
local nw = rowsof(W_weights)
forval i = 1/`nw' {
    if W_weights[`i',2] > 0.01 {
        file write fout "  Bang " %3.0f (W_weights[`i',1]) ///
            ": trong so = " %6.4f (W_weights[`i',2]) _n
    }
}
file write fout "================================================================" _n
file write fout "Phuong phap: synth2 voi nested + allopt optimization." _n
file write fout "Inference: Randomization inference (khong co SE thong thuong)." _n
file write fout "P-value hop le duy nhat: in-space placebo test." _n

file close fout
di as text "-> Da luu: ketqua/bang_scm_tong_hop.txt"


/*--------------------------------------------------------------------------
  TONG KET
--------------------------------------------------------------------------*/

di as text _newline "=== HOAN THANH ==="
di as text "{hline 65}"
di as text "  KET QUA DA LUU: thu muc ketqua/"
di as text "{hline 65}"
di as text "  BANG (TXT):"
di as text "    bang_x_balance.txt      — Kiem tra chat luong fit"
di as text "    bang_scm_tong_hop.txt   — Tong hop ket qua + W weights"
di as text _newline "  BIEU DO (PNG):"
di as text "    bieu_do_xu_huong_nen.png   — Raw trends truoc SCM"
di as text "    bieu_do_spaghetti.png      — Tat ca bang trong dong"
di as text "    bieu_do_scm_chinh.png      — CA vs Synthetic CA"
di as text "    bieu_do_gap_plot.png       — Gap (tac dong theo thoi gian)"
di as text "    bieu_do_placebo_inspace.png — In-space placebo test"
di as text "    bieu_do_placebo_intime.png  — In-time placebo test"
di as text "    bieu_do_loo.png             — Leave-one-out robustness"
di as text "{hline 65}"
di as text _newline "  TOM TAT:"
di as text "  RMSPE (1970-1988)  = " as result %6.3f rmspe_pre
di as text "  ATT 1989-2000      = " as result %6.1f att_full as text " bao/nam/1000"
di as text "  % giam             = " as result %5.1f abs(pct_red) "%"
di as text "  P-value (placebo)  = " as result %5.3f pval_inspace
di as text "{hline 65}"

/*==========================================================================
  GHI CHU PHUONG PHAP CHO NGHIEN CUU SINH

  1. TAI SAO SCM TOT HON DiD TRONG TRUONG HOP NAY?
     DiD yeu cau parallel trends giua CA va TUNG bang doi chung.
     SCM khong yeu cau dieu do — no tim to hop tuyen tinh toi uu
     sao cho khop lich su CA → linh hoat va minh bach hon.

  2. RMSPE LA TIEU CHI DANH GIA CHAT LUONG FIT:
     RMSPE truoc can thiep nho → synthetic CA khop CA thuc tot
     → ket qua sau can thiep dang tin cay hon.

  3. P-VALUE DUNG NHAT CHO SCM LA RANDOMIZATION INFERENCE:
     Chi co 1 don vi treated → khong the tinh SE thong thuong.
     P-value tu in-space placebo = ty le bang trong dong co gap
     lon bang hoac lon hon CA. P < 0.05 → ket qua co y nghia.

  4. TRONG SO W CO Y NGHIA KINH TE:
     Cac bang co W cao la nhung bang "giong California nhat"
     — tuc la counterfactual cua CA la to hop cua cac bang nay.
     Bao cao W weights la mot phan quan trong cua ket qua SCM.

  5. LOO KIEM TRA TINH VUNG:
     Neu mot bang co W rat lon (vi du 0.8), ket qua co the phu thuoc
     qua muc vao bang do. LOO kiem tra dieu nay bang cach loai
     lan luot tung bang co W > 0 va chay lai SCM.
==========================================================================*/
