#!/usr/bin/env python3
"""Build cinematic D3 dashboard — using .replace() for safe JS generation"""
import json, pandas as pd, numpy as np, os
from datetime import datetime

BASE = os.path.dirname(os.path.abspath(__file__))
OUTPUT_HTML = os.path.join(BASE, "output", "dashboard.html")

import sys
sys.path.insert(0, BASE)
from momentum_timing.strategy import SlopeTimingStrategy
from momentum_timing.data import fetch_index_data

# Get raw index data and run strategy for indicator values
raw = fetch_index_data('sh.000300', '2005-01-01', use_cache=True, verbose=False)
strategy = SlopeTimingStrategy(n_regression=16, m_standardize=300, threshold=0.7,
                                method='right_biased', use_price_filter=True)
result_full = strategy.compute(raw, verbose=False)

# Merge indicator with equity CSV
df = pd.read_csv(os.path.join(BASE, "output", "equity.csv"), parse_dates=["date"])
df = df.merge(result_full[['date', 'indicator']], on='date', how='left')

with open(os.path.join(BASE, "output", "summary.json"), encoding="utf-8") as f:
    summary = json.load(f)

perf = summary["策略表现"]; bench = summary["基准表现"]
trades_info = summary["交易统计"]; config = summary["策略配置"]
trades_list = trades_info.get("交易明细", [])

# Prep data
eq_data = []
for i in range(0, len(df), 4):
    r = df.iloc[i]
    eq_data.append([r["date"].strftime("%Y-%m-%d"), round(float(r["equity_curve_net"]), 3),
                     round(float(r["benchmark_curve"]), 3), round(float(r["drawdown"]) * 100, 1), int(r["pos"])])

# Annual
temp = df.copy(); temp["year"] = temp["date"].dt.year; temp["month"] = temp["date"].dt.month
monthly = temp.groupby(["year", "month"])["strategy_return"].apply(lambda x: (1 + x).prod() - 1).unstack()
annual = []
for y in monthly.index:
    v = monthly.loc[y].dropna()
    if len(v) > 0: annual.append([int(y), round(float((1 + v).prod() - 1) * 100, 1)])

# Heatmap
for m in range(1, 13):
    if m not in monthly.columns: monthly[m] = np.nan
monthly = monthly[sorted(monthly.columns)]
hm_years = monthly.index.astype(int).tolist()
hm_data = [[round(float(v) * 100, 1) if not np.isnan(v) else None for v in row] for row in monthly.values]

# Trades
trades_data = [[t["开始日期"], t["结束日期"], t["持仓天数"], t["收益率(%)"]] for t in trades_list]

# Trade distribution histogram bins
trade_rets = [t["收益率(%)"] for t in trades_list]
hist_bins = np.histogram(trade_rets, bins=10)
trade_hist = [[round(float(hist_bins[1][i]), 1), round(float(hist_bins[1][i+1]), 1), int(hist_bins[0][i])] for i in range(len(hist_bins[0]))]

# Cumulative trade returns
trade_cum = []; cum = 0.0
for i, t in enumerate(trades_list):
    cum += t["收益率(%)"]
    trade_cum.append([i + 1, t["收益率(%)"], round(cum, 2)])

# Holding days distribution
days_list = [t["持仓天数"] for t in trades_list]
days_bins = np.histogram(days_list, bins=10)
days_hist = [[round(float(days_bins[1][i])), round(float(days_bins[1][i+1])), int(days_bins[0][i])] for i in range(len(days_bins[0]))]

# Indicator data (sampled every 8 rows)
ind_data = []
sample_step = 8
for i in range(0, len(df), sample_step):
    r = df.iloc[i]
    v = float(r.get("indicator", 0))
    ind_data.append([r["date"].strftime("%Y-%m-%d"), round(v, 4) if not np.isnan(v) else 0, int(r["pos"])])

# Rolling
w = 252
roll_ann = []; roll_dd = []
for i in range(w - 1, len(df)):
    seg = df["strategy_return"].iloc[i - w + 1 : i + 1].values
    ann = ((1 + seg).prod() ** (1 / (w / 252)) - 1) * 100
    roll_ann.append([df["date"].iloc[i].strftime("%Y-%m-%d"), round(float(ann), 1)])
    eq = df["equity_curve_net"].iloc[i - w + 1 : i + 1].values
    pk = np.maximum.accumulate(eq)
    roll_dd.append([df["date"].iloc[i].strftime("%Y-%m-%d"), round(float(np.min((eq - pk) / pk) * 100), 1)])

data_js = "var _D={"
data_js += "eq:" + json.dumps(eq_data) + ","
data_js += "annual:" + json.dumps(annual) + ","
data_js += "hm:{years:" + json.dumps(hm_years) + ",data:" + json.dumps(hm_data) + "},"
data_js += "trades:" + json.dumps(trades_data) + ","
data_js += "rollAnn:" + json.dumps(roll_ann) + ","
data_js += "rollDD:" + json.dumps(roll_dd) + ","
data_js += "tradeHist:" + json.dumps(trade_hist) + ","
data_js += "tradeCum:" + json.dumps(trade_cum) + ","
data_js += "daysHist:" + json.dumps(days_hist) + ","
data_js += "indicator:" + json.dumps(ind_data) + ","
data_js += "perf:" + json.dumps(perf, ensure_ascii=False) + ","
data_js += "bench:" + json.dumps(bench, ensure_ascii=False) + ","
data_js += "tStats:" + json.dumps(trades_info, ensure_ascii=False) + ","
data_js += "config:" + json.dumps(config, ensure_ascii=False) + ","
data_js += "excess:" + json.dumps(summary.get("超额收益(%)", 0)) + ","
data_js += "dr:" + json.dumps(summary["数据日期"], ensure_ascii=False)
data_js += "};"

# ---- JS template (no f-string, injected via .replace) ----
JS_TEMPLATE = """
// ---- Utils ----
var fmt=function(n,d){return (n||0).toFixed(d||1);};
var fmtP=function(n){return (n>=0?'+':'')+fmt(n,1)+'%';};
var pd=d3.timeParse("%Y-%m-%d");
var $=function(s){return document.querySelector(s);};
var $$=function(s){return Array.from(document.querySelectorAll(s));};

// ---- KPI Bar ----
(function(){
  var items=[
    {v:fmt(_D.perf["累计收益率(%)"],0)+'%',l:'累计收益'},
    {v:fmtP(_D.perf["年化收益率(%)"]),l:'年化收益'},
    {v:fmt(_D.perf["夏普比率"],2),l:'夏普比率'},
    {v:fmt(_D.perf["最大回撤(%)"],1)+'%',l:'最大回撤'},
    {v:String(_D.tStats["交易次数"]),l:'交易次数'},
    {v:fmt(_D.tStats["胜率(%)"],1)+'%',l:'胜率'},
    {v:fmt(_D.tStats["盈亏比"],2),l:'盈亏比'},
    {v:fmtP(_D.excess),l:'超额收益'},
  ];
  var h='';
  items.forEach(function(it,i){
    if(i>0)h+='<span class="kd"></span>';
    h+='<div class="ki"><div class="kv">'+it.v+'</div><div class="kl">'+it.l+'</div></div>';
  });
  document.getElementById('kpiInner').innerHTML=h;
})();

// ---- Stats Grid ----
(function(){
  var d=[
    {v:fmt(_D.perf["累计收益率(%)"],0)+'%',l:'策略累计收益',n:'基准 '+fmt(_D.bench["累计收益率(%)"],0)+'%',c:'accent'},
    {v:fmtP(_D.perf["年化收益率(%)"]),l:'策略年化收益',n:'基准 '+fmtP(_D.bench["年化收益率(%)"]),c:'accent'},
    {v:fmt(_D.perf["夏普比率"],2),l:'夏普比率',n:'越高越好',c:'white'},
    {v:fmt(_D.perf["最大回撤(%)"],1)+'%',l:'最大回撤',n:'基准 '+fmt(_D.bench["最大回撤(%)"],1)+'%',c:'red'},
    {v:String(_D.tStats["交易次数"]),l:'交易次数',n:'胜率 '+fmt(_D.tStats["胜率(%)"],1)+'%',c:'white'},
    {v:fmt(_D.tStats["盈亏比"],2),l:'盈亏比',n:'均盈 '+fmt(_D.tStats["平均盈利(%)"],1)+'%',c:'green'},
    {v:_D.tStats["平均持仓天数"]+'天',l:'平均持仓天数',n:'',c:'white'},
    {v:fmtP(_D.excess),l:'超额收益',n:'vs 买入持有',c:'accent'},
  ];
  var h='';
  d.forEach(function(s){
    h+='<div class="sc"><div class="sv '+(s.c==='accent'?'gold':s.c==='red'?'rd':s.c==='green'?'gr':'')+'">'+s.v+'</div><div class="sl">'+s.l+'</div><div class="sn">'+s.n+'</div></div>';
  });
  document.getElementById('statsGrid').innerHTML=h;
})();

// ---- Config Chips ----
(function(){
  var cfg=_D.config;
  var items=[
    {l:'指标方法',v:cfg["策略方法"]},
    {l:'回归周期 N',v:cfg["回归周期(N)"]},
    {l:'标准分周期 M',v:cfg["标准分周期(M)"]},
    {l:'信号阈值 S',v:cfg["信号阈值"]},
    {l:'价格过滤',v:cfg["价格过滤"]},
    {l:'成交量过滤',v:cfg["成交量过滤"]},
  ];
  var h='';
  items.forEach(function(c){
    h+='<div class="cc"><span class="cd"></span><span class="cl">'+c.l+'</span><span class="cv">'+c.v+'</span></div>';
  });
  document.getElementById('configChips').innerHTML=h;
})();

// ---- Chart: Equity Curve ----
(function(){
  var container=document.getElementById('chartEquity');
  var data=_D.eq;
  var m={t:20,r:40,b:40,l:55};
  var W=Math.max(container.clientWidth||900,900)-m.l-m.r, H=500;
  var svg=d3.select(container).append('svg').attr('viewBox','0 0 '+(W+m.l+m.r)+' '+(H+m.t+m.b)).attr('class','chart-svg').style('min-height','500px');
  var g=svg.append('g').attr('transform','translate('+m.l+','+m.t+')');
  var x=d3.scaleTime().range([0,W]);
  var y=d3.scaleLog().range([H,0]).domain([0.5,d3.max(data,function(d){return Math.max(d[1],d[2])})*1.1]);
  x.domain(d3.extent(data,function(d){return pd(d[0])}));
  // clip
  g.append('defs').append('clipPath').attr('id','eqClip').append('rect').attr('width',W).attr('height',H);
  // benchmark
  var benchLine=d3.line().x(function(d){return x(pd(d[0]));}).y(function(d){return y(d[2]);});
  g.append('path').datum(data).attr('d',benchLine).attr('fill','none').attr('stroke','rgba(255,255,255,0.1)').attr('stroke-width',1.5).attr('stroke-dasharray','4,4');
  // strategy area
  var sg=svg.append('defs').append('linearGradient').attr('id','eqSG');
  sg.append('stop').attr('offset','0%').attr('stop-color','#d4a562').attr('stop-opacity',0.25);
  sg.append('stop').attr('offset','100%').attr('stop-color','#d4a562').attr('stop-opacity',0.02);
  var stratArea=d3.area().x(function(d){return x(pd(d[0]));}).y0(y(1)).y1(function(d){return y(d[1]);});
  g.append('path').datum(data).attr('d',stratArea).attr('fill','url(#eqSG)').attr('clip-path','url(#eqClip)');
  // strategy line
  var stratLine=d3.line().x(function(d){return x(pd(d[0]));}).y(function(d){return y(d[1]);});
  g.append('path').datum(data).attr('d',stratLine).attr('fill','none').attr('stroke','#d4a562').attr('stroke-width',2.2);
  // buy/sell markers
  var buys=[],sells=[];
  for(var i=1;i<data.length;i++){
    if(data[i][4]===1&&data[i-1][4]===0)buys.push(data[i]);
    if(data[i][4]===0&&data[i-1][4]===1)sells.push(data[i]);
  }
  g.selectAll('.bd').data(buys).enter().append('polygon').attr('points','0,-7 6,4 -6,4')
    .attr('transform',function(d){return 'translate('+x(pd(d[0]))+','+y(d[1])+')';}).attr('fill','#34d399').attr('opacity',0.9);
  g.selectAll('.sd').data(sells).enter().append('polygon').attr('points','0,7 6,-4 -6,-4')
    .attr('transform',function(d){return 'translate('+x(pd(d[0]))+','+y(d[1])+')';}).attr('fill','#f87171').attr('opacity',0.9);
  // axes
  g.append('g').attr('transform','translate(0,'+H+')').call(d3.axisBottom(x).ticks(10).tickFormat(d3.timeFormat('%Y')));
  g.append('g').call(d3.axisLeft(y).ticks(6).tickFormat(function(d){return d>=1?d.toFixed(0)+'x':d.toFixed(2);}));
  // tooltip
  var tip=d3.select(container).append('div').attr('class','d3t');
  var fLine=g.append('line').attr('stroke','rgba(255,255,255,0.15)').attr('y1',0).attr('y2',H).style('display','none');
  var fDot=g.append('circle').attr('r',4).attr('fill','#d4a562').style('display','none');
  var bisect=d3.bisector(function(d){return pd(d[0]);}).left;
  g.append('rect').attr('width',W).attr('height',H).attr('fill','none').attr('pointer-events','all')
    .on('mousemove',function(e){
      var mx=d3.pointer(e)[0];
      var dx=x.invert(mx);
      var i=bisect(data,dx,1);if(i>=data.length)return;
      var d=data[i];
      fLine.style('display',null).attr('x1',x(pd(d[0]))).attr('x2',x(pd(d[0])));
      fDot.style('display',null).attr('cx',x(pd(d[0]))).attr('cy',y(d[1]));
      tip.style('opacity',1).style('left',(x(pd(d[0]))+m.l+15)+'px').style('top',(y(d[1])+m.t-40)+'px')
        .html('<div class="ttd">'+d[0]+'</div><div class="ttv">净值 '+fmt(d[1],3)+'</div><div class="tts">基准 '+fmt(d[2],3)+' | 回撤 '+fmt(d[3],1)+'%</div>');
    })
    .on('mouseleave',function(){fLine.style('display','none');fDot.style('display','none');tip.style('opacity',0);});
})();

// ---- Chart: Drawdown ----
(function(){
  var container=document.getElementById('chartDrawdown');
  var data=_D.eq;
  var m={t:20,r:20,b:30,l:45};
  var W=Math.max(container.clientWidth||350,350)-m.l-m.r, H=200;
  var svg=d3.select(container).append('svg').attr('viewBox','0 0 '+(W+m.l+m.r)+' '+(H+m.t+m.b)).attr('class','chart-svg').style('min-height','200px');
  var g=svg.append('g').attr('transform','translate('+m.l+','+m.t+')');
  var x=d3.scaleTime().domain(d3.extent(data,function(d){return pd(d[0]);})).range([0,W]);
  var y=d3.scaleLinear().domain([d3.min(data,function(d){return d[3];})*1.1,0]).range([H,0]);
  var area=d3.area().x(function(d){return x(pd(d[0]));}).y0(y(0)).y1(function(d){return y(d[3]);});
  var dg=svg.append('defs').append('linearGradient').attr('id','ddG');
  dg.append('stop').attr('offset','0%').attr('stop-color','#f87171').attr('stop-opacity',0.3);
  dg.append('stop').attr('offset','100%').attr('stop-color','#f87171').attr('stop-opacity',0.02);
  g.append('path').datum(data).attr('d',area).attr('fill','url(#ddG)');
  var line=d3.line().x(function(d){return x(pd(d[0]));}).y(function(d){return y(d[3]);});
  g.append('path').datum(data).attr('d',line).attr('fill','none').attr('stroke','#f87171').attr('stroke-width',1.5);
  g.append('g').attr('transform','translate(0,'+H+')').call(d3.axisBottom(x).ticks(5).tickFormat(d3.timeFormat('%Y')));
  g.append('g').call(d3.axisLeft(y).ticks(4).tickFormat(function(d){return d+'%';}));
  // hover tip
  var tip=d3.select(container).append('div').attr('class','d3t');
  container.addEventListener('mousemove',function(e){
    var r=container.getBoundingClientRect();
    var mx=e.clientX-r.left-m.l; if(mx<0||mx>W){tip.style('opacity',0);return;}
    var dx=x.invert(mx);
    var bf=d3.bisector(function(d){return pd(d[0]);}).left;
    var i=bf(data,dx,1);if(i>=data.length)i=data.length-1;
    var d=data[i];
    tip.style('opacity',1).style('left',(e.clientX-r.left+15)+'px').style('top',(e.clientY-r.top-40)+'px')
      .html('<div class="ttd">'+d[0]+'</div><div class="ttv rd">回撤 '+fmt(d[3],1)+'%</div>');
  });
  container.addEventListener('mouseleave',function(){tip.style('opacity',0);});
})();

// ---- Chart: Indicator ----
(function(){
  var container=document.getElementById('chartIndicator');
  var data=_D.indicator;
  var m={t:20,r:20,b:30,l:45};
  var W=Math.max(container.clientWidth||350,350)-m.l-m.r, H=200;
  var svg=d3.select(container).append('svg').attr('viewBox','0 0 '+(W+m.l+m.r)+' '+(H+m.t+m.b)).attr('class','chart-svg').style('min-height','200px');
  var g=svg.append('g').attr('transform','translate('+m.l+','+m.t+')');
  var x=d3.scaleTime().domain(d3.extent(data,function(d){return pd(d[0]);})).range([0,W]);
  var vals=data.map(function(d){return d[1];}).filter(function(v){return isFinite(v);});
  var yM=Math.max(Math.abs(d3.max(vals)||2),Math.abs(d3.min(vals)||2))*1.3;
  var y=d3.scaleLinear().domain([-yM,yM]).range([H,0]);
  // pos regions
  var inP=null;
  data.forEach(function(d){
    if(d[2]===1&&!inP)inP={s:pd(d[0])};
    else if(d[2]===0&&inP){g.append('rect').attr('x',x(inP.s)).attr('width',Math.max(x(pd(d[0]))-x(inP.s),2)).attr('y',0).attr('height',H).attr('fill','rgba(52,211,153,0.04)');inP=null;}
  });
  if(inP)g.append('rect').attr('x',x(inP.s)).attr('width',Math.max(x(pd(data[data.length-1][0]))-x(inP.s),2)).attr('y',0).attr('height',H).attr('fill','rgba(52,211,153,0.04)');
  // indicator line
  var dline=d3.line().x(function(d){return x(pd(d[0]));}).y(function(d){return y(d[1]);});
  g.append('path').datum(data).attr('d',dline).attr('fill','none').attr('stroke','#22d3ee').attr('stroke-width',1.5);
  // thresholds
  g.append('line').attr('x1',0).attr('x2',W).attr('y1',y(0.7)).attr('y2',y(0.7)).attr('stroke','rgba(52,211,153,0.3)').attr('stroke-dasharray','4,3');
  g.append('line').attr('x1',0).attr('x2',W).attr('y1',y(-0.7)).attr('y2',y(-0.7)).attr('stroke','rgba(248,113,113,0.3)').attr('stroke-dasharray','4,3');
  g.append('line').attr('x1',0).attr('x2',W).attr('y1',y(0)).attr('y2',y(0)).attr('stroke','rgba(255,255,255,0.08)');
  g.append('g').attr('transform','translate(0,'+H+')').call(d3.axisBottom(x).ticks(5).tickFormat(d3.timeFormat('%Y')));
  g.append('g').call(d3.axisLeft(y).ticks(4));
  // hover tip
  var tip=d3.select(container).append('div').attr('class','d3t');
  container.addEventListener('mousemove',function(e){
    var r=container.getBoundingClientRect();
    var mx=e.clientX-r.left-m.l;if(mx<0||mx>W){tip.style('opacity',0);return;}
    var dx=x.invert(mx);
    var bf=d3.bisector(function(d){return pd(d[0]);}).left;
    var i=bf(data,dx,1);if(i>=data.length)i=data.length-1;
    var d=data[i];
    tip.style('opacity',1).style('left',(e.clientX-r.left+15)+'px').style('top',(e.clientY-r.top-40)+'px')
      .html('<div class="ttd">'+d[0]+'</div><div class="ttv">指标 '+fmt(d[1],3)+'</div><div class="tts">持仓: '+(d[2]?'是':'否')+'</div>');
  });
  container.addEventListener('mouseleave',function(){tip.style('opacity',0);});
})();

// ---- Chart: Annual Bar ----
(function(){
  var container=document.getElementById('chartAnnualBar');
  var data=_D.annual;
  var m={t:20,r:20,b:40,l:55};
  var W=Math.max(container.clientWidth||600,data.length*50)-m.l-m.r, H=240;
  var svg=d3.select(container).append('svg').attr('viewBox','0 0 '+(W+m.l+m.r)+' '+(H+m.t+m.b)).attr('class','chart-svg').style('min-height','300px');
  var g=svg.append('g').attr('transform','translate('+m.l+','+m.t+')');
  var x=d3.scaleBand().domain(data.map(function(d){return d[0];})).range([0,W]).padding(0.3);
  var yM=d3.max(data,function(d){return Math.abs(d[1]);})*1.2;
  var y=d3.scaleLinear().domain([-yM,yM]).range([H,0]);
  g.selectAll('rect').data(data).enter().append('rect')
    .attr('x',function(d){return x(d[0]);}).attr('width',x.bandwidth())
    .attr('y',function(d){return y(Math.max(0,d[1]));})
    .attr('height',function(d){return Math.abs(y(d[1])-y(0));})
    .attr('fill',function(d){return d[1]>=0?'#34d399':'#f87171';}).attr('rx',3).attr('opacity',0.8);
  g.selectAll('.vlb').data(data).enter().append('text')
    .attr('x',function(d){return x(d[0])+x.bandwidth()/2;})
    .attr('y',function(d){return d[1]>=0?y(d[1])-6:y(d[1])+14;})
    .attr('text-anchor','middle').attr('fill','#7a8599').style('font-size','10px')
    .text(function(d){return fmtP(d[1]);});
  g.append('line').attr('x1',0).attr('x2',W).attr('y1',y(0)).attr('y2',y(0)).attr('stroke','rgba(255,255,255,0.1)');
  g.append('g').attr('transform','translate(0,'+H+')').call(d3.axisBottom(x));
  g.append('g').call(d3.axisLeft(y).ticks(5).tickFormat(function(d){return d+'%';}));
  // hover
  var bars=$$('#chartAnnualBar rect');
  var tip=d3.select(container).append('div').attr('class','d3t');
  bars.forEach(function(bar,i){
    bar.addEventListener('mouseenter',function(e){
      bars.forEach(function(b){d3.select(b).transition().duration(250).attr('opacity',0.2);});
      d3.select(this).transition().duration(200).attr('opacity',1).attr('filter','brightness(1.3)');
      var d=data[i];
      var r=container.getBoundingClientRect();
      tip.style('opacity',1).style('left',(e.clientX-r.left+15)+'px').style('top',(e.clientY-r.top-50)+'px')
        .html('<div class="ttd">'+d[0]+'年</div><div class="ttv">'+fmtP(d[1])+'</div>');
    });
    bar.addEventListener('mouseleave',function(){
      bars.forEach(function(b){d3.select(b).transition().duration(350).attr('opacity',0.8).attr('filter',null);});
      tip.style('opacity',0);
    });
  });
})();

// ---- Chart: Trade Distribution Histogram ----
(function(){
  var container=document.getElementById('chartTradeDist');
  var data=_D.tradeHist;
  var m={t:20,r:20,b:30,l:45};
  var W=Math.max(container.clientWidth||350,350)-m.l-m.r, H=200;
  var svg=d3.select(container).append('svg').attr('viewBox','0 0 '+(W+m.l+m.r)+' '+(H+m.t+m.b)).attr('class','chart-svg').style('min-height','200px');
  var g=svg.append('g').attr('transform','translate('+m.l+','+m.t+')');
  var x=d3.scaleLinear().domain([d3.min(data,function(d){return d[0];}),d3.max(data,function(d){return d[1];})]).range([0,W]);
  var y=d3.scaleLinear().domain([0,d3.max(data,function(d){return d[2];})*1.2]).range([H,0]);
  g.selectAll('rect').data(data).enter().append('rect')
    .attr('x',function(d){return x(d[0]);}).attr('width',function(d){return Math.max(x(d[1])-x(d[0])-1,2);})
    .attr('y',function(d){return y(d[2]);}).attr('height',function(d){return H-y(d[2]);})
    .attr('fill',function(d){return (d[0]+d[1])/2>=0?'#34d399':'#f87171';}).attr('rx',3).attr('opacity',0.8);
  g.append('g').attr('transform','translate(0,'+H+')').call(d3.axisBottom(x).ticks(6).tickFormat(function(d){return d+'%';}));
  g.append('g').call(d3.axisLeft(y).ticks(4));
  var tip=d3.select(container).append('div').attr('class','d3t');
  g.selectAll('rect').on('mouseenter',function(e,d){
    d3.select(this).transition().duration(150).attr('opacity',1).attr('filter','brightness(1.3)');
    tip.style('opacity',1).html('<div class="ttv">'+d[2]+'笔</div><div class="tts">'+fmtP(d[0])+' ~ '+fmtP(d[1])+'</div>')
      .style('left',(e.offsetX+15)+'px').style('top',(e.offsetY-50)+'px');
  }).on('mousemove',function(e){
    tip.style('left',(e.offsetX+15)+'px').style('top',(e.offsetY-50)+'px');
  }).on('mouseleave',function(){
    d3.select(this).transition().duration(200).attr('opacity',0.8).attr('filter',null);
    tip.style('opacity',0);
  });
})();

// ---- Chart: Cumulative Trade Return ----
(function(){
  var container=document.getElementById('chartTradeCum');
  var data=_D.tradeCum;
  var m={t:20,r:20,b:30,l:55};
  var W=Math.max(container.clientWidth||350,350)-m.l-m.r, H=200;
  var svg=d3.select(container).append('svg').attr('viewBox','0 0 '+(W+m.l+m.r)+' '+(H+m.t+m.b)).attr('class','chart-svg').style('min-height','200px');
  var g=svg.append('g').attr('transform','translate('+m.l+','+m.t+')');
  var x=d3.scaleLinear().domain([1,data.length]).range([0,W]);
  var yM=d3.max(data,function(d){return d[2];})*1.15;
  var ym=d3.min(data,function(d){return d[2];});
  var y=d3.scaleLinear().domain([Math.min(ym*1.2,-5),yM]).range([H,0]);
  // waterfall bars
  var bw=Math.min(W/data.length*0.7,18);
  g.selectAll('rect').data(data).enter().append('rect')
    .attr('x',function(d){return x(d[0])-bw/2;}).attr('width',bw)
    .attr('y',function(d){return y(Math.max(0,d[1]));})
    .attr('height',function(d){return Math.abs(y(d[1])-y(0));})
    .attr('fill',function(d){return d[1]>=0?'#34d399':'#f87171';}).attr('rx',2).attr('opacity',0.75);
  // cumulative line
  var cumLine=d3.line().x(function(d){return x(d[0]);}).y(function(d){return y(d[2]);}).curve(d3.curveMonotoneX);
  g.append('path').datum(data).attr('d',cumLine).attr('fill','none').attr('stroke','#d4a562').attr('stroke-width',2);
  g.append('line').attr('x1',0).attr('x2',W).attr('y1',y(0)).attr('y2',y(0)).attr('stroke','rgba(255,255,255,0.1)');
  g.append('g').attr('transform','translate(0,'+H+')').call(d3.axisBottom(x).ticks(8).tickFormat(function(d){return '#'+d;}));
  g.append('g').call(d3.axisLeft(y).ticks(5).tickFormat(function(d){return d+'%';}));
  var tip=d3.select(container).append('div').attr('class','d3t');
  g.selectAll('rect').on('mouseenter',function(e,d){
    d3.select(this).transition().duration(150).attr('opacity',1).attr('filter','brightness(1.3)');
    tip.style('opacity',1).html('<div class="ttd">交易 #'+d[0]+'</div><div class="ttv">'+fmtP(d[1])+'</div><div class="tts">累计 '+fmt(d[2],1)+'%</div>')
      .style('left',(e.offsetX+15)+'px').style('top',(e.offsetY-60)+'px');
  }).on('mouseleave',function(){
    d3.select(this).transition().duration(200).attr('opacity',0.75).attr('filter',null);
    tip.style('opacity',0);
  });
})();

// ---- Chart: Holding Days Distribution ----
(function(){
  var container=document.getElementById('chartDaysDist');
  var data=_D.daysHist;
  var m={t:20,r:20,b:30,l:45};
  var W=Math.max(container.clientWidth||350,350)-m.l-m.r, H=200;
  var svg=d3.select(container).append('svg').attr('viewBox','0 0 '+(W+m.l+m.r)+' '+(H+m.t+m.b)).attr('class','chart-svg').style('min-height','200px');
  var g=svg.append('g').attr('transform','translate('+m.l+','+m.t+')');
  var x=d3.scaleLinear().domain([0,d3.max(data,function(d){return d[1];})*1.05]).range([0,W]);
  var y=d3.scaleLinear().domain([0,d3.max(data,function(d){return d[2];})*1.2]).range([H,0]);
  g.selectAll('rect').data(data).enter().append('rect')
    .attr('x',function(d){return x(d[0]);}).attr('width',function(d){return Math.max(x(d[1])-x(d[0])-1,2);})
    .attr('y',function(d){return y(d[2]);}).attr('height',function(d){return H-y(d[2]);})
    .attr('fill','#60a5fa').attr('rx',3).attr('opacity',0.7);
  g.append('g').attr('transform','translate(0,'+H+')').call(d3.axisBottom(x).ticks(6).tickFormat(function(d){return d+'天';}));
  g.append('g').call(d3.axisLeft(y).ticks(4));
  var tip=d3.select(container).append('div').attr('class','d3t');
  g.selectAll('rect').on('mouseenter',function(e,d){
    d3.select(this).transition().duration(150).attr('opacity',1).attr('filter','brightness(1.2)');
    tip.style('opacity',1).html('<div class="ttv">'+d[2]+'笔</div><div class="tts">'+Math.round(d[0])+'~'+Math.round(d[1])+'天</div>')
      .style('left',(e.offsetX+15)+'px').style('top',(e.offsetY-50)+'px');
  }).on('mousemove',function(e){
    tip.style('left',(e.offsetX+15)+'px').style('top',(e.offsetY-50)+'px');
  }).on('mouseleave',function(){
    d3.select(this).transition().duration(200).attr('opacity',0.7).attr('filter',null);
    tip.style('opacity',0);
  });
})();

// ---- Chart: Rolling Annual Return ----
(function(){
  var container=document.getElementById('chartRollRet');
  var data=_D.rollAnn.filter(function(_,i){return i%3===0;});
  var m={t:20,r:20,b:30,l:50};
  var W=Math.max(container.clientWidth||350,350)-m.l-m.r, H=200;
  var svg=d3.select(container).append('svg').attr('viewBox','0 0 '+(W+m.l+m.r)+' '+(H+m.t+m.b)).attr('class','chart-svg').style('min-height','200px');
  var g=svg.append('g').attr('transform','translate('+m.l+','+m.t+')');
  var x=d3.scaleTime().domain(d3.extent(data,function(d){return pd(d[0]);})).range([0,W]);
  var yM=d3.max(data,function(d){return d[1];})*1.2,ym=d3.min(data,function(d){return d[1];});
  var y=d3.scaleLinear().domain([Math.min(ym*1.3,-10),Math.max(yM,10)]).range([H,0]);
  var area=d3.area().x(function(d){return x(pd(d[0]));}).y0(y(0)).y1(function(d){return y(d[1]);});
  g.append('path').datum(data).attr('d',area).attr('fill','rgba(96,165,250,0.1)');
  var line=d3.line().x(function(d){return x(pd(d[0]));}).y(function(d){return y(d[1]);});
  g.append('path').datum(data).attr('d',line).attr('fill','none').attr('stroke','#60a5fa').attr('stroke-width',1.5);
  g.append('line').attr('x1',0).attr('x2',W).attr('y1',y(0)).attr('y2',y(0)).attr('stroke','rgba(255,255,255,0.1)');
  g.append('g').attr('transform','translate(0,'+H+')').call(d3.axisBottom(x).ticks(5).tickFormat(d3.timeFormat('%Y')));
  g.append('g').call(d3.axisLeft(y).ticks(4).tickFormat(function(d){return d+'%';}));
  var tip=d3.select(container).append('div').attr('class','d3t');
  container.addEventListener('mousemove',function(e){
    var r=container.getBoundingClientRect();
    var mx=e.clientX-r.left-m.l;if(mx<0||mx>W){tip.style('opacity',0);return;}
    var dx=x.invert(mx);
    var bf=d3.bisector(function(d){return pd(d[0]);}).left;
    var i=bf(data,dx,1);if(i>=data.length)i=data.length-1;
    var d=data[i];
    tip.style('opacity',1).style('left',(e.clientX-r.left+15)+'px').style('top',(e.clientY-r.top-40)+'px')
      .html('<div class="ttd">'+d[0]+'</div><div class="ttv">'+fmtP(d[1])+' 年化</div>');
  });
  container.addEventListener('mouseleave',function(){tip.style('opacity',0);});
})();

// ---- Chart: Rolling Max Drawdown ----
(function(){
  var container=document.getElementById('chartRollDD');
  var data=_D.rollDD.filter(function(_,i){return i%3===0;});
  var m={t:20,r:20,b:30,l:50};
  var W=Math.max(container.clientWidth||350,350)-m.l-m.r, H=200;
  var svg=d3.select(container).append('svg').attr('viewBox','0 0 '+(W+m.l+m.r)+' '+(H+m.t+m.b)).attr('class','chart-svg').style('min-height','200px');
  var g=svg.append('g').attr('transform','translate('+m.l+','+m.t+')');
  var x=d3.scaleTime().domain(d3.extent(data,function(d){return pd(d[0]);})).range([0,W]);
  var y=d3.scaleLinear().domain([d3.min(data,function(d){return d[1];})*1.1,0]).range([H,0]);
  var area=d3.area().x(function(d){return x(pd(d[0]));}).y0(y(0)).y1(function(d){return y(d[1]);});
  g.append('path').datum(data).attr('d',area).attr('fill','rgba(248,113,113,0.12)');
  var line=d3.line().x(function(d){return x(pd(d[0]));}).y(function(d){return y(d[1]);});
  g.append('path').datum(data).attr('d',line).attr('fill','none').attr('stroke','#f87171').attr('stroke-width',1.5);
  g.append('g').attr('transform','translate(0,'+H+')').call(d3.axisBottom(x).ticks(5).tickFormat(d3.timeFormat('%Y')));
  g.append('g').call(d3.axisLeft(y).ticks(4).tickFormat(function(d){return d+'%';}));
  var tip=d3.select(container).append('div').attr('class','d3t');
  container.addEventListener('mousemove',function(e){
    var r=container.getBoundingClientRect();
    var mx=e.clientX-r.left-m.l;if(mx<0||mx>W){tip.style('opacity',0);return;}
    var dx=x.invert(mx);
    var bf=d3.bisector(function(d){return pd(d[0]);}).left;
    var i=bf(data,dx,1);if(i>=data.length)i=data.length-1;
    var d=data[i];
    tip.style('opacity',1).style('left',(e.clientX-r.left+15)+'px').style('top',(e.clientY-r.top-40)+'px')
      .html('<div class="ttd">'+d[0]+'</div><div class="ttv rd">回撤 '+fmt(d[1],1)+'%</div>');
  });
  container.addEventListener('mouseleave',function(){tip.style('opacity',0);});
})();

// ---- Trade Table ----
(function(){
  var tbody=document.getElementById('tradeTBody');
  var trades=_D.trades;
  function render(filter){
    var f=filter||function(){return true;};
    tbody.innerHTML=trades.filter(f).map(function(t,i){
      return '<tr class="'+(t[3]>=0?'win':'loss')+'"><td style="color:#4a5568">#'+(i+1)+'</td><td>'+t[0]+'</td><td>'+t[1]+'</td><td>'+t[2]+'天</td><td>'+fmtP(t[3])+'</td></tr>';
    }).join('');
  }
  render();
  // filter buttons
  document.getElementById('tradeAll').addEventListener('click',function(){render();actBtn(this,'.tfb');});
  document.getElementById('tradeWin').addEventListener('click',function(){render(function(t){return t[3]>0;});actBtn(this,'.tfb');});
  document.getElementById('tradeLoss').addEventListener('click',function(){render(function(t){return t[3]<=0;});actBtn(this,'.tfb');});
  // export
  document.getElementById('tradeExp').addEventListener('click',function(){
    var rows=tbody.querySelectorAll('tr');
    var csv='序号,开始日期,结束日期,持仓天数,收益率\\n';
    rows.forEach(function(r,i){
      var c=r.querySelectorAll('td');
      csv+=(i+1)+','+c[1].textContent+','+c[2].textContent+','+c[3].textContent+','+c[4].textContent+'\\n';
    });
    var blob=new Blob(['\\uFEFF'+csv],{type:'text/csv;charset=utf-8'});
    var url=URL.createObjectURL(blob);
    var a=document.createElement('a');a.href=url;a.download='交易记录.csv';a.click();
    URL.revokeObjectURL(url);
  });
  function actBtn(btn,sel){$$(sel).forEach(function(b){b.classList.remove('active');});btn.classList.add('active');}
})();

// ---- Heatmap ----
(function(){
  var container=document.getElementById('chartHeatmap');
  var hm=_D.hm;
  var data=hm.data, years=hm.years;
  var cs=22, gap=3;
  var W=12*(cs+gap), H=data.length*(cs+gap);
  var maxA=d3.max(data.flat().filter(function(v){return v!=null;}),function(d){return Math.abs(d);})||20;
  var svg=d3.select(container).append('svg').attr('viewBox','0 0 '+(W+70)+' '+(H+20)).attr('class','chart-svg').style('min-height',(H+30)+'px');
  var g=svg.append('g').attr('transform','translate(50,10)');
  var cs2=d3.scaleSequential(d3.interpolateRdYlGn).domain([maxA,-maxA]);
  data.forEach(function(row,ri){
    row.forEach(function(val,ci){
      if(val===null)return;
      g.append('rect').attr('x',ci*(cs+gap)).attr('y',ri*(cs+gap)).attr('width',cs).attr('height',cs).attr('rx',2)
        .attr('fill',cs2(val)).append('title').text(years[ri]+'年'+(ci+1)+'月: '+fmtP(val));
    });
  });
  data.forEach(function(_,ri){if(ri%2===0)g.append('text').attr('x',-8).attr('y',ri*(cs+gap)+cs/2+4).attr('text-anchor','end').attr('fill','#4a5568').style('font-size','10px').text(years[ri]);});
  for(var ci=0;ci<12;ci++)g.append('text').attr('x',ci*(cs+gap)+cs/2).attr('y',-4).attr('text-anchor','middle').attr('fill','#4a5568').style('font-size','8px').text(ci+1);
  // filters
  document.getElementById('hmAll').addEventListener('click',function(){
    $$('#chartHeatmap rect').forEach(function(r){r.style.opacity='';});actBtn2(this,'.hmb');
  });
  document.getElementById('hmBull').addEventListener('click',function(){
    $$('#chartHeatmap rect').forEach(function(r,i){
      var row=Math.floor(i/12);r.style.opacity=(years[row]>=2006&&years[row]<=2007)?'':'0.08';
    });actBtn2(this,'.hmb');
  });
  document.getElementById('hmBear').addEventListener('click',function(){
    $$('#chartHeatmap rect').forEach(function(r,i){
      var row=Math.floor(i/12);r.style.opacity=(years[row]===2008)?'':'0.08';
    });actBtn2(this,'.hmb');
  });
  function actBtn2(btn,sel){$$(sel).forEach(function(b){b.classList.remove('active');});btn.classList.add('active');}
  // hover tooltip
  var tip=d3.select(container).append('div').attr('class','d3t');
  $$('#chartHeatmap rect').forEach(function(cell){
    var te=cell.querySelector('title');
    var tt=te?te.textContent:'';
    cell.addEventListener('mouseenter',function(e){
      var r=container.getBoundingClientRect();
      tip.style('opacity',1).html('<div class="ttd">'+tt+'</div>').style('left',(e.clientX-r.left+15)+'px').style('top',(e.clientY-r.top-40)+'px');
    });
    cell.addEventListener('mouseleave',function(){tip.style('opacity',0);});
  });
})();
"""

# Read the HTML skeleton
with open(__file__, "r") as f:
    pass  # We'll build it directly

# ---- Build complete HTML ----
css_part = """
*,*::before,*::after{box-sizing:border-box;margin:0;padding:0}
:root{--bg:#080c14;--bg2:rgba(14,18,28,0.85);--bg3:rgba(20,26,40,0.7);--bd:rgba(255,255,255,0.06);--t1:#e8ecf1;--t2:#7a8599;--t3:#4a5568;--gold:#d4a562;--gr:#34d399;--rd:#f87171;--cy:#22d3ee;--bl:#60a5fa;--fd:Georgia,Times New Roman,serif;--fb:-apple-system,BlinkMacSystemFont,Segoe UI,PingFang SC,Microsoft YaHei,sans-serif;--fm:SF Mono,Cascadia Code,JetBrains Mono,monospace}
html{scroll-behavior:smooth;background:var(--bg);color:var(--t1);font-family:var(--fb);font-size:16px;line-height:1.6;-webkit-font-smoothing:antialiased}
body{overflow-x:hidden}
::-webkit-scrollbar{width:6px}
::-webkit-scrollbar-track{background:var(--bg)}
::-webkit-scrollbar-thumb{background:var(--t3);border-radius:3px}

.hero{position:relative;z-index:1;min-height:100vh;display:flex;flex-direction:column;align-items:center;justify-content:center;padding:60px 40px;text-align:center}
.hero-eyebrow{font-family:var(--fm);font-size:0.75rem;letter-spacing:0.35em;text-transform:uppercase;color:var(--gold);margin-bottom:24px}
.hero-title{font-family:var(--fd);font-size:clamp(2.5rem,6vw,5rem);font-weight:400;line-height:1.1;letter-spacing:-0.02em;margin-bottom:16px}
.hero-title em{font-style:normal;color:var(--gold)}
.hero-subtitle{font-size:1.1rem;color:var(--t2);max-width:600px;margin-bottom:48px}
.hero-subtitle em{color:var(--gold);font-style:normal}
.hero-meta{display:flex;gap:32px;font-family:var(--fm);font-size:0.8rem;color:var(--t3)}
.hero-meta span{display:flex;align-items:center;gap:8px}
.hero-meta .dot{width:6px;height:6px;border-radius:50%;background:var(--gold)}

.kpi-bar{position:sticky;top:0;z-index:100;background:rgba(8,12,20,0.92);backdrop-filter:blur(20px);-webkit-backdrop-filter:blur(20px);border-bottom:1px solid var(--bd);padding:14px 0}
.kpi-bar-inner{max-width:1400px;margin:0 auto;padding:0 40px;display:flex;gap:0;justify-content:space-between;align-items:center}
.ki{text-align:center;flex:1}
.kv{font-family:var(--fd);font-size:1.6rem;font-weight:400;color:var(--t1);line-height:1.2;letter-spacing:-0.01em}
.kl{font-size:0.7rem;color:var(--t3);text-transform:uppercase;letter-spacing:0.1em;margin-top:2px}
.kd{width:1px;height:30px;background:var(--bd)}

.container{position:relative;z-index:1;max-width:1400px;margin:0 auto;padding:0 40px}
.section{margin-bottom:80px;padding-top:20px}
.section-header{margin-bottom:40px}
.section-label{font-family:var(--fm);font-size:0.7rem;letter-spacing:0.25em;text-transform:uppercase;color:var(--gold);margin-bottom:8px}
.section-title{font-family:var(--fd);font-size:2rem;font-weight:400;letter-spacing:-0.01em;color:var(--t1)}
.section-desc{font-size:0.95rem;color:var(--t2);max-width:600px;margin-top:8px}

.chart-panel{background:var(--bg2);border:1px solid var(--bd);border-radius:16px;padding:32px;margin-bottom:24px;backdrop-filter:blur(10px);-webkit-backdrop-filter:blur(10px);position:relative;overflow:hidden;transition:border-color 0.4s,box-shadow 0.4s}
.chart-panel:hover{border-color:rgba(212,165,98,0.15);box-shadow:0 0 60px rgba(212,165,98,0.04)}
.chart-panel::before{content:'';position:absolute;top:0;left:0;right:0;height:1px;background:linear-gradient(90deg,transparent,rgba(212,165,98,0.15),transparent);opacity:0.5}
.cpt{font-family:var(--fd);font-size:1.15rem;color:var(--t1);margin-bottom:20px;letter-spacing:-0.01em}
.cpd{font-size:0.8rem;color:var(--t3);margin-bottom:24px}
.chart-svg{width:100%;display:block}

.stats-grid{display:grid;grid-template-columns:repeat(auto-fit,minmax(220px,1fr));gap:16px;margin-bottom:24px}
.sc{background:var(--bg3);border:1px solid var(--bd);border-radius:12px;padding:20px;position:relative;overflow:hidden;transition:transform 0.35s,border-color 0.35s,box-shadow 0.35s}
.sc:hover{transform:translateY(-4px);border-color:rgba(212,165,98,0.25);box-shadow:0 12px 40px rgba(0,0,0,0.3),0 0 0 1px rgba(212,165,98,0.1) inset}
.sv{font-family:var(--fd);font-size:2rem;font-weight:400;letter-spacing:-0.02em;line-height:1.1;transition:transform 0.3s}
.sc:hover .sv{transform:scale(1.05)}
.sv.gold{color:var(--gold)}.sv.rd{color:var(--rd)}.sv.gr{color:var(--gr)}
.sl{font-size:0.75rem;color:var(--t3);margin-top:4px;text-transform:uppercase;letter-spacing:0.08em}
.sn{font-size:0.7rem;color:var(--t3);margin-top:6px;opacity:0.7;transition:opacity 0.3s}
.sc:hover .sn{opacity:1}

.grid-2{display:grid;grid-template-columns:1fr 1fr;gap:24px}
@media(max-width:900px){.grid-2{grid-template-columns:1fr}}

.config-row{display:flex;flex-wrap:wrap;gap:10px;margin-bottom:24px}
.cc{display:inline-flex;align-items:center;gap:8px;padding:8px 16px;background:var(--bg3);border:1px solid var(--bd);border-radius:100px;font-size:0.8rem;font-family:var(--fm);transition:all 0.3s;cursor:default}
.cc:hover{border-color:var(--gold);background:rgba(212,165,98,0.08);transform:translateY(-1px)}
.cd{width:6px;height:6px;border-radius:50%;background:var(--gold);transition:transform 0.3s}
.cc:hover .cd{transform:scale(1.5)}
.cl{color:var(--t3);font-size:0.7rem}
.cv{color:var(--t1)}

.trade-table{width:100%;border-collapse:collapse;font-size:0.85rem}
.trade-table th{text-align:left;padding:12px 16px;color:var(--t3);font-weight:500;font-size:0.75rem;text-transform:uppercase;letter-spacing:0.08em;border-bottom:1px solid var(--bd);position:sticky;top:0;background:var(--bg2);z-index:2}
.trade-table td{padding:10px 16px;border-bottom:1px solid rgba(255,255,255,0.03);font-family:var(--fm);font-size:0.8rem}
.trade-table tbody tr:nth-child(even) td{background:rgba(255,255,255,0.01)}
.trade-table tr.win td{color:var(--gr)}
.trade-table tr.loss td{color:var(--rd)}
.trade-table tbody tr:hover td{background:rgba(212,165,98,0.06)!important}
.trade-scroll{max-height:500px;overflow-y:auto;border-radius:0 0 16px 16px}

.filter-bar{display:flex;align-items:center;gap:16px;flex-wrap:wrap;margin-bottom:16px}
.filter-label{font-size:0.75rem;color:var(--t3);text-transform:uppercase;letter-spacing:0.08em}
.fbtn{background:var(--bg3);border:1px solid var(--bd);border-radius:8px;color:var(--t2);padding:8px 16px;font-size:0.8rem;cursor:pointer;font-family:var(--fb);transition:all 0.2s}
.fbtn:hover{border-color:var(--gold);color:var(--gold)}
.fbtn.active{background:var(--gold);color:var(--bg);border-color:var(--gold)}
.export-btn{margin-left:auto;display:flex;align-items:center;gap:6px}

.d3t{position:absolute;background:rgba(20,26,40,0.96);border:1px solid rgba(255,255,255,0.12);border-radius:8px;padding:12px 16px;pointer-events:none;font-size:0.8rem;opacity:0;transition:opacity 0.15s;z-index:200;backdrop-filter:blur(8px);box-shadow:0 8px 32px rgba(0,0,0,0.4)}
.ttd{color:var(--t3);font-size:0.7rem;margin-bottom:4px}
.ttv{color:var(--t1);font-family:var(--fd);font-size:1.1rem}
.ttv.rd{color:var(--rd)}
.tts{color:var(--t2);font-size:0.75rem;margin-top:2px}

.footer{text-align:center;padding:60px 40px;border-top:1px solid var(--bd);margin-top:60px}
.footer-text{font-size:0.8rem;color:var(--t3)}
.disclaimer{background:rgba(251,191,36,0.06);border:1px solid rgba(251,191,36,0.15);border-radius:12px;padding:20px 24px;font-size:0.8rem;color:#b49450;line-height:1.6;margin-top:32px}

@media(prefers-reduced-motion:reduce){.sc:hover{transform:none}}
"""

html_content = f"""<!DOCTYPE html>
<html lang="zh-CN">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>量化择时策略 · 回测仪表盘</title>
<script src="https://d3js.org/d3.v7.min.js"></script>
<style>{css_part}</style>
</head>
<body>

<section class="hero">
  <div class="hero-eyebrow">量化择时策略 · 21年回测验证</div>
  <h1 class="hero-title">量价斜率<br><em>择时系统</em></h1>
  <p class="hero-subtitle">沪深300指数 · 右偏标准分 + 价格过滤 · 累计收益 <em>{perf['累计收益率(%)']:.0f}%</em></p>
  <div class="hero-meta">
    <span><span class="dot"></span>数据：baostock</span>
    <span><span class="dot"></span>区间：{summary['数据日期']['开始']} — {summary['数据日期']['结束']}</span>
    <span><span class="dot"></span>标的：沪深300</span>
  </div>
</section>

<div class="kpi-bar"><div class="kpi-bar-inner" id="kpiInner"></div></div>

<div class="container">

  <section class="section" id="sec-overview">
    <div class="section-header">
      <div class="section-label">01 · 绩效概览</div>
      <div class="section-title">策略 vs 基准</div>
      <div class="section-desc">年化 {perf['年化收益率(%)']:.1f}% vs {bench['年化收益率(%)']:.1f}%，最大回撤 {abs(perf['最大回撤(%)']):.1f}% vs {abs(bench['最大回撤(%)']):.1f}%</div>
    </div>
    <div class="config-row" id="configChips"></div>
    <div class="stats-grid" id="statsGrid"></div>
    <div class="chart-panel">
      <div class="cpt">净值曲线（对数坐标）· 策略 vs 基准</div>
      <div class="cpd">金色 = 策略净值 | 灰虚线 = 买入持有 | ▲ 绿 = 买入 | ▼ 红 = 卖出</div>
      <div id="chartEquity" style="position:relative"></div>
    </div>
  </section>

  <section class="section" id="sec-analysis">
    <div class="section-header">
      <div class="section-label">02 · 信号与回撤</div>
      <div class="section-title">择时指标与市场状态</div>
    </div>
    <div class="grid-2">
      <div class="chart-panel"><div class="cpt">择时指标</div><div class="cpd">青色线 = 右偏标准分指标 | 虚线 = ±0.7 阈值 | 绿底色 = 持仓区间</div><div id="chartIndicator" style="position:relative"></div></div>
      <div class="chart-panel"><div class="cpt">回撤分析</div><div id="chartDrawdown" style="position:relative"></div></div>
    </div>
    <div class="chart-panel" style="margin-top:24px">
      <div class="cpt">月度收益热力图</div>
      <div class="cpd">绿 = 盈利 | 红 = 亏损 | 深浅 = 幅度</div>
      <div class="filter-bar">
        <span class="filter-label">筛选</span>
        <button class="fbtn active hmb" id="hmAll">全部</button>
        <button class="fbtn hmb" id="hmBull">牛市 06-07</button>
        <button class="fbtn hmb" id="hmBear">熊市 2008</button>
      </div>
      <div id="chartHeatmap" style="position:relative;overflow-x:auto"></div>
    </div>
  </section>

  <section class="section" id="sec-trade-analysis">
    <div class="section-header">
      <div class="section-label">03 · 交易分析</div>
      <div class="section-title">每笔交易拆解</div>
      <div class="section-desc">{trades_info['交易次数']} 笔交易，{trades_info['盈利次数']} 盈 {trades_info['亏损次数']} 亏，胜率 {trades_info['胜率(%)']:.1f}%，盈亏比 {trades_info['盈亏比']:.2f}</div>
    </div>
    <div class="grid-2">
      <div class="chart-panel"><div class="cpt">交易收益分布</div><div id="chartTradeDist" style="position:relative"></div></div>
      <div class="chart-panel"><div class="cpt">累计交易收益</div><div id="chartTradeCum" style="position:relative"></div></div>
    </div>
    <div class="grid-2" style="margin-top:24px">
      <div class="chart-panel"><div class="cpt">持仓天数分布</div><div id="chartDaysDist" style="position:relative"></div></div>
      <div class="chart-panel"><div class="cpt">各年度收益率</div><div id="chartAnnualBar" style="position:relative"></div></div>
    </div>
  </section>

  <section class="section" id="sec-rolling">
    <div class="section-header">
      <div class="section-label">04 · 滚动性能</div>
      <div class="section-title">策略在不同市场环境的表现</div>
      <div class="section-desc">252日滚动窗口，观察策略年化收益和最大回撤的时变特征</div>
    </div>
    <div class="grid-2">
      <div class="chart-panel"><div class="cpt">滚动年化收益率（252日）</div><div id="chartRollRet" style="position:relative"></div></div>
      <div class="chart-panel"><div class="cpt">滚动最大回撤（252日）</div><div id="chartRollDD" style="position:relative"></div></div>
    </div>
  </section>

  <section class="section" id="sec-trades">
    <div class="section-header">
      <div class="section-label">05 · 交易明细</div>
      <div class="section-title">全部 {trades_info['交易次数']} 笔交易</div>
    </div>
    <div class="chart-panel" style="padding:24px 0 0 0">
      <div style="display:flex;align-items:center;padding:0 24px 16px">
        <span class="filter-label">筛选</span>
        <button class="fbtn active tfb" id="tradeAll" style="margin-left:12px">全部</button>
        <button class="fbtn tfb" id="tradeWin" style="margin-left:8px">盈利 ({trades_info['盈利次数']})</button>
        <button class="fbtn tfb" id="tradeLoss" style="margin-left:8px">亏损 ({trades_info['亏损次数']})</button>
        <button class="fbtn export-btn" id="tradeExp" style="margin-left:auto">⬇ 导出 CSV</button>
      </div>
      <div class="trade-scroll"><table class="trade-table"><thead><tr><th>#</th><th>开始</th><th>结束</th><th>持仓</th><th>收益</th></tr></thead><tbody id="tradeTBody"></tbody></table></div>
    </div>
  </section>

  <div class="footer">
    <p class="footer-text">生成时间：{datetime.now().strftime('%Y-%m-%d %H:%M')} · 数据源：baostock · 标的：沪深300指数</p>
    <div class="disclaimer">⚠️ 以上内容由量化策略模型基于历史数据模拟生成，仅供学习研究参考，不构成任何投资建议。回测收益不代表未来表现。</div>
  </div>

</div>

<script>
{data_js}
{JS_TEMPLATE}
</script>
</body>
</html>"""

os.makedirs(os.path.dirname(OUTPUT_HTML), exist_ok=True)
with open(OUTPUT_HTML, "w", encoding="utf-8") as f:
    f.write(html_content)

size_kb = os.path.getsize(OUTPUT_HTML) / 1024
print(f"✅ {OUTPUT_HTML} ({size_kb:.0f} KB)")

# Verify brace balance
with open(OUTPUT_HTML, "r") as f:
    html = f.read()
# Extract inline JS
s = html.find("<script>", html.find("var _D="))
e = html.find("</script>", s)
js = html[s+8:e]
bd = 0; pd = 0
for ch in js:
    if ch == '{': bd += 1
    if ch == '}': bd -= 1
    if ch == '(': pd += 1
    if ch == ')': pd -= 1
print(f"Brace balance: {bd} {'✅' if bd == 0 else '❌'}")
print(f"Paren balance: {pd} {'✅' if pd == 0 else '❌'}")
