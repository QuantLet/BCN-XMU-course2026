* 1. 清空环境并设置工作目录
clear all
cd "D:\e盘数据\区块链作业\ETH&BTC&其他国货币对美元汇率15分钟1小时1天"  

* 2. 第一步：处理所有资产的时间格式（统一提取纯日期，剔除00:00:00）
* 定义循环处理函数，避免重复代码（处理所有5个资产）
capture program drop process_data
program define process_data
    args input_file output_file var_name
    import delimited "`input_file'", clear varnames(1)
    keep time close
    rename close `var_name'  // 重命名收盘价变量（如ETHUSD、BTCUSD）
    * 提取纯日期（剔除空格后的00:00:00）
    gen date_str = substr(time, 1, strpos(time, " ") - 1)
    gen date_stata = date(date_str, "YMD")
    format date_stata %td
    drop time  // 删除原始带时间后缀的变量
    save "`output_file'", replace
end

* 调用函数处理每个资产数据
process_data "ETHUSD_D1.csv" "ETH_processed.dta" "ETHUSD"  // ETH（基准）
process_data "BTCUSD_D1.csv" "BTC_processed.dta" "BTCUSD"  // BTC
process_data "AUDUSD_D1.csv" "AUD_processed.dta" "AUDUSD"  // 澳元
process_data "EURUSD_D1.csv" "EUR_processed.dta" "EURUSD"  // 欧元
process_data "GBPUSD_D1.csv" "GBP_processed.dta" "GBPUSD"  // 英镑

* 3. 第二步：多轮merge，仅保留所有资产完全匹配的日期
* 以ETH为基础，逐步合并其他资产，每步均剔除未匹配项
use "ETH_processed.dta", clear  // 初始数据集：仅ETH
* 合并BTC，只保留双方都有的日期（match）
merge 1:1 date_stata using "BTC_processed.dta", keep(match) nogenerate
* 合并澳元，剔除未匹配项
merge 1:1 date_stata using "AUD_processed.dta", keep(match) nogenerate
* 合并欧元，剔除未匹配项
merge 1:1 date_stata using "EUR_processed.dta", keep(match) nogenerate
* 合并英镑，剔除未匹配项
merge 1:1 date_stata using "GBP_processed.dta", keep(match) nogenerate

**********************************************************************
tsset date_stata  // 用处理好的纯日期变量作为时间索引

* 1. 计算日度对数收益率（滞后项基于匹配后的时间序列）
gen ret_ETH = log(ETHUSD) - log(L1.ETHUSD)  // ETH收益率
gen ret_BTC = log(BTCUSD) - log(L1.BTCUSD)  // BTC收益率
gen ret_AUD = log(AUDUSD) - log(L1.AUDUSD)  // 澳元兑美元收益率
gen ret_EUR = log(EURUSD) - log(L1.EURUSD)  // 欧元兑美元收益率
gen ret_GBP = log(GBPUSD) - log(L1.GBPUSD)  // 英镑兑美元收益率

* 2. 删除因滞后项产生的首行缺失值
drop if missing(ret_ETH, ret_BTC, ret_AUD, ret_EUR, ret_GBP)

sort date_stata

* 2. 创建连续的时间索引
gen t = _n

* 3. 设置连续时间序列
tsset t

* 4.预测条件方差 & 协方差

mgarch dcc  ///
(ret_ETH = ) ///
(ret_BTC = ) ///
(ret_AUD = ) ///
(ret_EUR = ) ///
(ret_GBP = ), ///
arch(1) garch(1) distribution(t)
predict h_*, variance
**********************************************************************
* 1. 方差变量
local var1_h = "h__ret_ETH_ret_ETH"   // ETH方差
local var2_h = "h__ret_BTC_ret_BTC"   // BTC方差
local var3_h = "h__ret_AUD_ret_AUD"   // AUD方差
local var4_h = "h__ret_EUR_ret_EUR"   // EUR方差
local var5_h = "h__ret_GBP_ret_GBP"   // GBP方差

* 2. 协方差变量
local cov_ETH_BTC = "h__ret_BTC_ret_ETH"
local cov_ETH_AUD = "h__ret_AUD_ret_ETH"
local cov_ETH_EUR = "h__ret_EUR_ret_ETH"
local cov_ETH_GBP = "h__ret_GBP_ret_ETH"
local cov_BTC_AUD = "h__ret_AUD_ret_BTC"
local cov_BTC_EUR = "h__ret_EUR_ret_BTC"
local cov_BTC_GBP = "h__ret_GBP_ret_BTC"

* ------------------------------------------------
* 3. 生成动态相关系数
* ------------------------------------------------
gen corr_ETH_BTC  = `cov_ETH_BTC' / sqrt(`var1_h' * `var2_h')
gen corr_ETH_AUD  = `cov_ETH_AUD' / sqrt(`var1_h' * `var3_h')
gen corr_ETH_EUR  = `cov_ETH_EUR' / sqrt(`var1_h' * `var4_h')
gen corr_ETH_GBP  = `cov_ETH_GBP' / sqrt(`var1_h' * `var5_h')

gen corr_BTC_AUD  = `cov_BTC_AUD' / sqrt(`var2_h' * `var3_h')
gen corr_BTC_EUR  = `cov_BTC_EUR' / sqrt(`var2_h' * `var4_h')
gen corr_BTC_GBP  = `cov_BTC_GBP' / sqrt(`var2_h' * `var5_h')

* 4.绘图
* ETH与澳元
tsline corr_ETH_AUD corr_BTC_AUD, ///
       title("加密货币与澳元的动态条件相关性", size(medium)) ///
       ytitle("动态相关系数") ///
       xtitle("时间") ///
       legend(pos(12) ring(0) cols(2) ///
              label(1 "ETH-AUD") label(2 "BTC-AUD")) ///
       ylabel(-0.4(0.2)0.8) ///
       lwidth(medium) ///
       scheme(s1color) ///
       name(corr_aud, replace)
	   
* ETH与欧元
tsline corr_ETH_EUR corr_BTC_EUR, ///
       title("加密货币与欧元的动态条件相关性", size(medium)) ///
       ytitle("动态相关系数") ///
       xtitle("时间") ///
       legend(pos(12) ring(0) cols(2) ///
              label(1 "ETH-EUR") label(2 "BTC-EUR")) ///
       ylabel(-0.4(0.2)0.8) ///
       lwidth(medium) ///
       scheme(s1color) ///
       name(corr_eur, replace)

* ETH与英镑
tsline corr_ETH_GBP corr_BTC_GBP, ///
       title("加密货币与英镑的动态条件相关性", size(medium)) ///
       ytitle("动态相关系数") ///
       xtitle("时间") ///
       legend(pos(12) ring(0) cols(2) ///
              label(1 "ETH-GBP") label(2 "BTC-GBP")) ///
       ylabel(-0.4(0.2)0.8) ///
       lwidth(medium) ///
       scheme(s1color) ///
       name(corr_gbp, replace)
	   
graph combine corr_aud corr_eur corr_gbp, ///
              rows(3) ///
              title("加密货币与法币动态条件相关性（DCC-GARCH）", size(medium)) ///
              scheme(s1color)
* 3. 疫情分段（2020-03-11）
*----------------------
* 定义疫情前后分段
*----------------------
gen period = .

replace period = 1 if date_stata < td(11mar2020)   // 疫情前
replace period = 2 if date_stata >= td(11mar2020)  // 疫情后

*----------------------
* 给 period 加标签
*----------------------
label define period_label 1 "疫情前" 2 "疫情后"
label values period period_label
*----------------------
* 绘图
*----------------------
graph box corr_ETH_AUD corr_BTC_AUD, ///
    over(period) ///
    ytitle("动态相关系数") ///
    legend(label(1 "ETH-AUD") label(2 "BTC-AUD")) ///
    scheme(s1color) ///
    name(box_aud, replace)

graph box corr_ETH_EUR corr_BTC_EUR, ///
    over(period) ///
    ytitle("动态相关系数") ///
    legend(label(1 "ETH-EUR") label(2 "BTC-EUR")) ///
    scheme(s1color) ///
    name(box_eur, replace)
	
graph box corr_ETH_GBP corr_BTC_GBP, ///
    over(period) ///
    ytitle("动态相关系数") ///
    legend(label(1 "ETH-GBP") label(2 "BTC-GBP")) ///
    scheme(s1color) ///
    name(box_gbp, replace)

graph combine box_aud box_eur box_gbp, ///
    rows(3) ///
    scheme(s1color)
* 4. 分段统计相关性（基于纯匹配数据）
tabstat corr_ETH_EUR corr_ETH_AUD corr_ETH_GBP, ///
        by(period) ///
        stats(mean sd max min p50) ///
        longstub column(statistics) ///
        title("ETH与传统货币相关性（疫情前后，仅匹配数据）")
