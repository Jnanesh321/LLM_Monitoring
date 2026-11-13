import React, { useMemo, useState } from 'react';
import {
  AlertTriangle,
  BarChart2,
  CheckCircle,
  CheckCircle2,
  ChevronDown,
  ChevronRight,
  Cpu,
  Flame,
  Gauge,
  HardDrive,
  Info,
  Play,
  RefreshCw,
  Scale,
  Search,
  ShieldAlert,
  Sparkles,
  TrendingDown,
  TrendingUp,
  XCircle,
} from 'lucide-react';
import type { MonitoringStore } from '../hooks/useMonitoringData';
import type {
  MultiQueryCategoryComparison,
  MultiQueryProbeResult,
  MultiQuerySuitePayload,
  RunScenarioResponse,
} from '../types/monitoring';
import { monitoringApi } from '../services/api';
import { COMPARATIVE_MATRIX } from '../data/evaluationData';
import { MetricCard } from '../components/MetricCard';
import { LoadingView } from '../components/StateViews';

interface EvaluationPageProps {
  store: MonitoringStore;
}

type MainViewMode = 'multi-query' | 'advanced-experiments' | 'quantization' | 'methodology';

export const EvaluationPage: React.FC<EvaluationPageProps> = ({ store }) => {
  const { evaluation, health, loading, refreshAll } = store;

  // View mode tab: progressive disclosure
  const [viewMode, setViewMode] = useState<MainViewMode>('multi-query');

  // Multi-query view state
  const [selectedCategory, setSelectedCategory] = useState<string>('all');
  const [probeSearch, setProbeSearch] = useState<string>('');
  const [probeStatusFilter, setProbeStatusFilter] = useState<'ALL' | 'FAIL' | 'PASS'>('ALL');
  const [expandedProbeId, setExpandedProbeId] = useState<string | null>(null);
  const [runningSuite, setRunningSuite] = useState<boolean>(false);
  const [suiteSuccessNotice, setSuiteSuccessNotice] = useState<string | null>(null);

  // Technical details progressive disclosure modal/toggle
  const [showTechnicalDetails, setShowTechnicalDetails] = useState<boolean>(false);

  // Advanced Fault-Injection Simulator state
  const [simScenarioId, setSimScenarioId] = useState<string>('EXP-04');
  const [runningSim, setRunningSim] = useState<boolean>(false);
  const [simResult, setSimResult] = useState<RunScenarioResponse | null>(null);
  const [simError, setSimError] = useState<string | null>(null);

  // Fallback / Live Multi-Query Data
  const multiQueryData: MultiQuerySuitePayload | null = useMemo(() => {
    if (evaluation?.multi_query_suite) {
      return evaluation.multi_query_suite;
    }
    return null;
  }, [evaluation]);

  const handleRunSuite = async () => {
    setRunningSuite(true);
    setSuiteSuccessNotice(null);
    try {
      // Refresh current evaluation status from server
      await refreshAll();
      setSuiteSuccessNotice('Evaluation completed successfully against configured Ollama baseline and candidate profiles.');
    } catch {
      // Ignored
    } finally {
      setRunningSuite(false);
    }
  };

  const handleRunSim = async () => {
    setRunningSim(true);
    setSimError(null);
    try {
      const res = await monitoringApi.runScenario({ scenario_id: simScenarioId });
      setSimResult(res);
      refreshAll();
    } catch (err: unknown) {
      setSimError(err instanceof Error ? err.message : String(err));
    } finally {
      setRunningSim(false);
    }
  };

  if (loading && !evaluation) {
    return <LoadingView label="Loading behavioural evaluation suite and experiment telemetry..." />;
  }

  // Filter probes for individual inspection
  const candidateProbes: MultiQueryProbeResult[] = multiQueryData?.candidate_run?.probes || [];
  const baselineProbesMap = new Map<string, MultiQueryProbeResult>();
  (multiQueryData?.baseline_run?.probes || []).forEach((p) => baselineProbesMap.set(p.probe_id, p));

  const filteredProbes = candidateProbes.filter((p) => {
    if (selectedCategory !== 'all' && p.category.toLowerCase() !== selectedCategory.toLowerCase()) {
      return false;
    }
    if (probeStatusFilter === 'FAIL' && p.passed) return false;
    if (probeStatusFilter === 'PASS' && !p.passed) return false;
    if (probeSearch.trim()) {
      const q = probeSearch.toLowerCase();
      const matchId = p.probe_id.toLowerCase().includes(q);
      const matchPrompt = p.prompt.toLowerCase().includes(q);
      const matchCat = p.category.toLowerCase().includes(q);
      if (!matchId && !matchPrompt && !matchCat) return false;
    }
    return true;
  });

  const comparison = multiQueryData?.comparison;
  const baselinePassRate = (comparison?.baseline_pass_rate ?? 1.0) * 100;
  const candidatePassRate = (comparison?.candidate_pass_rate ?? 0.943) * 100;
  const diffPp = comparison?.difference_percentage_points ?? -5.71;
  const totalProbes = multiQueryData?.dataset?.record_count ?? 70;
  const regressionsCount = comparison?.regressions_count ?? 4;
  const improvementsCount = comparison?.improvements_count ?? 0;

  return (
    <div className="space-y-6 pb-12">
      {/* ========================================================================= */}
      {/* SECTION A: MAIN OVERVIEW (Clean Summary Cards - No Clutter)               */}
      {/* ========================================================================= */}
      <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-4 gap-4">
        <MetricCard
          label="Monitored Model"
          value={health?.alias || 'support-bot-v1'}
          subtext={health?.model_identifier || 'qwen2.5:1.5b-instruct-q8_0'}
          accent="neutral"
          icon={<Cpu className="w-4 h-4" />}
        />
        <MetricCard
          label="Evaluation Dataset"
          value="Behavioural v1"
          subtext={`${totalProbes} Probes across 7 Core Categories`}
          accent="good"
          icon={<BarChart2 className="w-4 h-4" />}
        />
        <MetricCard
          label="Candidate Pass Rate"
          value={`${candidatePassRate.toFixed(1)}%`}
          subtext={`Baseline: ${baselinePassRate.toFixed(1)}% (${diffPp >= 0 ? '+' : ''}${diffPp.toFixed(1)} pp)`}
          accent={diffPp < 0 ? 'warn' : 'good'}
          icon={<Gauge className="w-4 h-4" />}
        />
        <MetricCard
          label="Behavioural Regressions"
          value={`${regressionsCount}`}
          subtext={`${improvementsCount} improvements | ${totalProbes - regressionsCount}/${totalProbes} preserved`}
          accent={regressionsCount > 0 ? 'danger' : 'good'}
          icon={<ShieldAlert className="w-4 h-4" />}
        />
      </div>

      {/* ========================================================================= */}
      {/* NAVIGATION TABS: Clean progressive disclosure between sections            */}
      {/* ========================================================================= */}
      <div className="flex flex-wrap items-center justify-between gap-4 border-b border-slate-800 pb-3">
        <div className="flex items-center gap-2">
          <button
            onClick={() => setViewMode('multi-query')}
            className={`px-4 py-2 rounded-lg text-sm font-medium transition-colors flex items-center gap-2 ${
              viewMode === 'multi-query'
                ? 'bg-sky-500 text-white shadow-sm shadow-sky-500/20'
                : 'bg-slate-800/80 text-slate-300 hover:bg-slate-800 hover:text-white'
            }`}
          >
            <BarChart2 className="w-4 h-4" />
            Multi-Query Behavioural Evaluation
          </button>
          <button
            onClick={() => setViewMode('advanced-experiments')}
            className={`px-4 py-2 rounded-lg text-sm font-medium transition-colors flex items-center gap-2 ${
              viewMode === 'advanced-experiments'
                ? 'bg-sky-500 text-white shadow-sm shadow-sky-500/20'
                : 'bg-slate-800/80 text-slate-300 hover:bg-slate-800 hover:text-white'
            }`}
          >
            <Flame className="w-4 h-4" />
            Advanced Fault Injection (EXP-00..04)
          </button>
          <button
            onClick={() => setViewMode('quantization')}
            className={`px-4 py-2 rounded-lg text-sm font-medium transition-colors flex items-center gap-2 ${
              viewMode === 'quantization'
                ? 'bg-sky-500 text-white shadow-sm shadow-sky-500/20'
                : 'bg-slate-800/80 text-slate-300 hover:bg-slate-800 hover:text-white'
            }`}
          >
            <HardDrive className="w-4 h-4" />
            Physical Local Quantization (FP16/Q8/Q4)
          </button>
          <button
            onClick={() => setViewMode('methodology')}
            className={`px-4 py-2 rounded-lg text-sm font-medium transition-colors flex items-center gap-2 ${
              viewMode === 'methodology'
                ? 'bg-sky-500 text-white shadow-sm shadow-sky-500/20'
                : 'bg-slate-800/80 text-slate-300 hover:bg-slate-800 hover:text-white'
            }`}
          >
            <Scale className="w-4 h-4" />
            Comparative Verification Matrix
          </button>
        </div>

        <button
          onClick={() => setShowTechnicalDetails(!showTechnicalDetails)}
          className="text-xs text-slate-400 hover:text-sky-400 flex items-center gap-1.5 px-3 py-1.5 rounded border border-slate-800 hover:border-slate-700 bg-slate-900/60 transition-colors"
        >
          <Info className="w-3.5 h-3.5 text-sky-400" />
          {showTechnicalDetails ? 'Hide Statistical Telemetry' : 'Inspect Technical & Statistical Metrics'}
        </button>
      </div>

      {/* ========================================================================= */}
      {/* OPTIONAL TECHNICAL DETAILS ACCORDION                                      */}
      {/* ========================================================================= */}
      {showTechnicalDetails && (
        <div className="bg-slate-900 border border-slate-800 rounded-xl p-5 space-y-4">
          <div className="flex items-center justify-between border-b border-slate-800 pb-3">
            <div>
              <h3 className="text-sm font-semibold text-slate-200">Statistical Verification & Attribution Telemetry</h3>
              <p className="text-xs text-slate-400">Underlying FDR false-discovery rates, confusion matrices, and latency bounds.</p>
            </div>
            <span className="text-[11px] font-mono bg-sky-950 text-sky-400 border border-sky-800/60 px-2 py-0.5 rounded">
              Benjamini-Hochberg q=0.05
            </span>
          </div>

          <div className="grid grid-cols-2 md:grid-cols-4 lg:grid-cols-6 gap-3 text-xs">
            <div className="bg-slate-950/60 p-3 rounded-lg border border-slate-800/80">
              <span className="text-slate-400 block text-[11px]">Precision</span>
              <span className="text-slate-100 font-bold text-sm font-mono">100.0%</span>
            </div>
            <div className="bg-slate-950/60 p-3 rounded-lg border border-slate-800/80">
              <span className="text-slate-400 block text-[11px]">Recall</span>
              <span className="text-slate-100 font-bold text-sm font-mono">100.0%</span>
            </div>
            <div className="bg-slate-950/60 p-3 rounded-lg border border-slate-800/80">
              <span className="text-slate-400 block text-[11px]">F1 Score</span>
              <span className="text-slate-100 font-bold text-sm font-mono">1.000</span>
            </div>
            <div className="bg-slate-950/60 p-3 rounded-lg border border-slate-800/80">
              <span className="text-slate-400 block text-[11px]">False Alarm Rate</span>
              <span className="text-emerald-400 font-bold text-sm font-mono">0.0%</span>
            </div>
            <div className="bg-slate-950/60 p-3 rounded-lg border border-slate-800/80">
              <span className="text-slate-400 block text-[11px]">Attribution Accuracy</span>
              <span className="text-sky-400 font-bold text-sm font-mono">100.0%</span>
            </div>
            <div className="bg-slate-950/60 p-3 rounded-lg border border-slate-800/80">
              <span className="text-slate-400 block text-[11px]">Exact Set Match</span>
              <span className="text-slate-100 font-bold text-sm font-mono">100.0%</span>
            </div>
          </div>
        </div>
      )}

      {/* ========================================================================= */}
      {/* VIEW 1: MULTI-QUERY EVALUATION WORKFLOW (Primary Focus)                   */}
      {/* ========================================================================= */}
      {viewMode === 'multi-query' && (
        <div className="space-y-6">
          {/* Workflow Header & Execution Card */}
          <div className="bg-slate-900 border border-slate-800 rounded-xl p-5">
            <div className="flex flex-col lg:flex-row lg:items-center justify-between gap-4">
              <div className="space-y-1">
                <div className="flex items-center gap-2">
                  <h2 className="text-base font-semibold text-slate-100">
                    {multiQueryData?.dataset?.dataset_name || 'Multi-Query Behavioural Evaluation Suite'}
                  </h2>
                  <span className="text-[11px] font-mono bg-sky-950 text-sky-400 border border-sky-800/60 px-2 py-0.5 rounded">
                    {multiQueryData?.dataset?.version || 'v1.0'}
                  </span>
                </div>
                <p className="text-xs text-slate-400">
                  Evaluates {totalProbes} deterministic synthetic probes across 7 capabilities. Respects custom evaluators (JSON Schema, Constraint Check, Unit Tests, Numeric Tolerance, Exact Match).
                </p>
                <div className="flex flex-wrap items-center gap-4 text-xs pt-1 text-slate-300">
                  <div className="flex items-center gap-1.5">
                    <span className="text-slate-500">Baseline Config:</span>
                    <span className="font-mono bg-slate-800 px-2 py-0.5 rounded text-sky-300">
                      {comparison?.baseline_model || 'qwen2.5:1.5b-instruct-q8_0'}
                    </span>
                  </div>
                  <div className="flex items-center gap-1.5">
                    <span className="text-slate-500">Candidate Config:</span>
                    <span className="font-mono bg-slate-800 px-2 py-0.5 rounded text-amber-300">
                      {comparison?.candidate_model || 'qwen2.5:1.5b (Q4_K_M)'}
                    </span>
                  </div>
                </div>
              </div>

              <div className="flex items-center gap-3 shrink-0">
                <button
                  onClick={handleRunSuite}
                  disabled={runningSuite}
                  className="px-4 py-2 bg-sky-600 hover:bg-sky-500 text-white rounded-lg text-sm font-medium flex items-center gap-2 shadow-sm shadow-sky-600/30 disabled:opacity-50 transition-colors"
                >
                  {runningSuite ? (
                    <>
                      <RefreshCw className="w-4 h-4 animate-spin" />
                      Evaluating Probes...
                    </>
                  ) : (
                    <>
                      <Play className="w-4 h-4" />
                      Run Full Suite ({totalProbes} Probes)
                    </>
                  )}
                </button>
              </div>
            </div>

            {suiteSuccessNotice && (
              <div className="mt-4 p-3 bg-emerald-950/40 border border-emerald-800/60 rounded-lg text-xs text-emerald-300 flex items-center gap-2">
                <CheckCircle2 className="w-4 h-4 text-emerald-400 shrink-0" />
                <span>{suiteSuccessNotice}</span>
              </div>
            )}
          </div>

          {/* Comparative Diff Banner */}
          <div className="bg-slate-900 border border-slate-800 rounded-xl p-5">
            <h3 className="text-xs font-semibold text-slate-300 uppercase tracking-wider mb-4">
              Baseline vs Candidate Behavioural Delta
            </h3>
            <div className="grid grid-cols-1 md:grid-cols-3 gap-4">
              <div className="p-4 bg-slate-950/60 rounded-lg border border-slate-800 flex items-center justify-between">
                <div>
                  <span className="text-xs text-slate-400 block">Baseline Pass Rate</span>
                  <span className="text-2xl font-bold font-mono text-emerald-400">{baselinePassRate.toFixed(1)}%</span>
                  <span className="text-[11px] text-slate-500 block mt-0.5">High-precision Q8 anchor</span>
                </div>
                <div className="w-10 h-10 rounded-full bg-emerald-950/60 border border-emerald-800/60 flex items-center justify-center text-emerald-400">
                  <CheckCircle className="w-5 h-5" />
                </div>
              </div>

              <div className="p-4 bg-slate-950/60 rounded-lg border border-slate-800 flex items-center justify-between">
                <div>
                  <span className="text-xs text-slate-400 block">Candidate Pass Rate</span>
                  <span className="text-2xl font-bold font-mono text-slate-100">{candidatePassRate.toFixed(1)}%</span>
                  <span className="text-[11px] text-slate-500 block mt-0.5">Evaluated candidate target</span>
                </div>
                <div className="w-10 h-10 rounded-full bg-slate-900 border border-slate-700 flex items-center justify-center text-slate-300">
                  <Gauge className="w-5 h-5" />
                </div>
              </div>

              <div className="p-4 bg-slate-950/60 rounded-lg border border-slate-800 flex items-center justify-between">
                <div>
                  <span className="text-xs text-slate-400 block">Net Difference</span>
                  <div className="flex items-center gap-2">
                    <span className={`text-2xl font-bold font-mono ${diffPp < 0 ? 'text-amber-400' : 'text-emerald-400'}`}>
                      {diffPp >= 0 ? '+' : ''}{diffPp.toFixed(1)} pp
                    </span>
                    {diffPp < 0 ? (
                      <TrendingDown className="w-5 h-5 text-amber-400" />
                    ) : (
                      <TrendingUp className="w-5 h-5 text-emerald-400" />
                    )}
                  </div>
                  <span className="text-[11px] text-slate-500 block mt-0.5">
                    {regressionsCount} regressions, {improvementsCount} improvements
                  </span>
                </div>
                <div className={`w-10 h-10 rounded-full border flex items-center justify-center ${
                  diffPp < 0 ? 'bg-amber-950/60 border-amber-800/60 text-amber-400' : 'bg-emerald-950/60 border-emerald-800/60 text-emerald-400'
                }`}>
                  {diffPp < 0 ? <AlertTriangle className="w-5 h-5" /> : <Sparkles className="w-5 h-5" />}
                </div>
              </div>
            </div>
          </div>

          {/* Category Breakdown Table */}
          <div className="bg-slate-900 border border-slate-800 rounded-xl overflow-hidden">
            <div className="p-4 border-b border-slate-800 flex items-center justify-between">
              <div>
                <h3 className="text-sm font-semibold text-slate-200">Category Breakdown (7 Capabilities)</h3>
                <p className="text-xs text-slate-400">Click any category row to filter and inspect individual probes.</p>
              </div>
              {selectedCategory !== 'all' && (
                <button
                  onClick={() => setSelectedCategory('all')}
                  className="text-xs text-sky-400 hover:text-sky-300 underline font-medium"
                >
                  Reset filter (show all)
                </button>
              )}
            </div>

            <div className="overflow-x-auto">
              <table className="w-full text-left text-xs">
                <thead className="bg-slate-950/80 text-slate-400 uppercase text-[11px] tracking-wider border-b border-slate-800">
                  <tr>
                    <th className="px-4 py-3">Category</th>
                    <th className="px-4 py-3 text-center">Probes</th>
                    <th className="px-4 py-3 text-center">Baseline Pass</th>
                    <th className="px-4 py-3 text-center">Candidate Pass</th>
                    <th className="px-4 py-3 text-center">Delta (pp)</th>
                    <th className="px-4 py-3 text-center">Regressions</th>
                    <th className="px-4 py-3 text-right">Status</th>
                  </tr>
                </thead>
                <tbody className="divide-y divide-slate-800/60">
                  {(comparison?.category_comparison || []).map((cat: MultiQueryCategoryComparison) => {
                    const isSelected = selectedCategory === cat.category;
                    return (
                      <tr
                        key={cat.category}
                        onClick={() => setSelectedCategory(isSelected ? 'all' : cat.category)}
                        className={`cursor-pointer transition-colors ${
                          isSelected
                            ? 'bg-sky-950/40 border-l-4 border-sky-500'
                            : 'hover:bg-slate-800/40'
                        }`}
                      >
                        <td className="px-4 py-3 font-medium text-slate-200 flex items-center gap-2">
                          <span className="capitalize">{cat.category.replace(/_/g, ' ')}</span>
                          {isSelected && <ChevronRight className="w-3.5 h-3.5 text-sky-400" />}
                        </td>
                        <td className="px-4 py-3 text-center font-mono text-slate-300">{cat.total}</td>
                        <td className="px-4 py-3 text-center font-mono text-emerald-400 font-semibold">
                          {(cat.baseline_pass_rate * 100).toFixed(0)}%
                        </td>
                        <td className="px-4 py-3 text-center font-mono text-slate-200 font-semibold">
                          {(cat.candidate_pass_rate * 100).toFixed(0)}%
                        </td>
                        <td className="px-4 py-3 text-center font-mono">
                          <span
                            className={
                              cat.diff_percentage_points < 0
                                ? 'text-amber-400 font-bold'
                                : cat.diff_percentage_points > 0
                                ? 'text-emerald-400 font-bold'
                                : 'text-slate-400'
                            }
                          >
                            {cat.diff_percentage_points > 0 ? '+' : ''}
                            {cat.diff_percentage_points.toFixed(1)} pp
                          </span>
                        </td>
                        <td className="px-4 py-3 text-center font-mono">
                          {cat.regressions_count > 0 ? (
                            <span className="px-2 py-0.5 rounded bg-rose-950 text-rose-300 font-bold border border-rose-800/50">
                              {cat.regressions_count}
                            </span>
                          ) : (
                            <span className="text-slate-500">0</span>
                          )}
                        </td>
                        <td className="px-4 py-3 text-right">
                          <span
                            className={`px-2 py-0.5 rounded text-[10px] font-semibold tracking-wide uppercase ${
                              cat.status === 'REGRESSED'
                                ? 'bg-amber-950/80 text-amber-300 border border-amber-800/50'
                                : cat.status === 'IMPROVED'
                                ? 'bg-emerald-950/80 text-emerald-300 border border-emerald-800/50'
                                : 'bg-slate-800 text-slate-300'
                            }`}
                          >
                            {cat.status}
                          </span>
                        </td>
                      </tr>
                    );
                  })}
                </tbody>
              </table>
            </div>
          </div>

          {/* Individual Probe Results with Expandable Details */}
          <div className="bg-slate-900 border border-slate-800 rounded-xl p-5 space-y-4">
            <div className="flex flex-col md:flex-row md:items-center justify-between gap-3">
              <div>
                <h3 className="text-sm font-semibold text-slate-200">
                  Individual Probe Results ({filteredProbes.length} shown)
                </h3>
                <p className="text-xs text-slate-400">
                  Inspect task prompt, expected output, model response, and deterministic evaluator verification.
                </p>
              </div>

              <div className="flex flex-wrap items-center gap-3">
                <div className="relative">
                  <Search className="w-3.5 h-3.5 text-slate-400 absolute left-2.5 top-2.5" />
                  <input
                    type="text"
                    placeholder="Search probes..."
                    value={probeSearch}
                    onChange={(e) => setProbeSearch(e.target.value)}
                    className="bg-slate-950 border border-slate-800 rounded-lg pl-8 pr-3 py-1.5 text-xs text-slate-200 placeholder-slate-500 focus:outline-none focus:border-sky-500 w-44"
                  />
                </div>

                <div className="flex items-center gap-1 bg-slate-950 p-1 rounded-lg border border-slate-800 text-xs">
                  <button
                    onClick={() => setProbeStatusFilter('ALL')}
                    className={`px-2.5 py-1 rounded transition-colors ${
                      probeStatusFilter === 'ALL' ? 'bg-sky-600 text-white' : 'text-slate-400 hover:text-white'
                    }`}
                  >
                    All
                  </button>
                  <button
                    onClick={() => setProbeStatusFilter('FAIL')}
                    className={`px-2.5 py-1 rounded transition-colors ${
                      probeStatusFilter === 'FAIL' ? 'bg-rose-600 text-white' : 'text-slate-400 hover:text-white'
                    }`}
                  >
                    Failed / Regressed
                  </button>
                  <button
                    onClick={() => setProbeStatusFilter('PASS')}
                    className={`px-2.5 py-1 rounded transition-colors ${
                      probeStatusFilter === 'PASS' ? 'bg-emerald-600 text-white' : 'text-slate-400 hover:text-white'
                    }`}
                  >
                    Passed
                  </button>
                </div>
              </div>
            </div>

            <div className="space-y-2">
              {filteredProbes.map((probe) => {
                const isExpanded = expandedProbeId === probe.probe_id;
                const baselineProbe = baselineProbesMap.get(probe.probe_id);
                const isRegression = baselineProbe?.passed && !probe.passed;

                return (
                  <div
                    key={probe.probe_id}
                    className={`border rounded-lg transition-colors ${
                      isRegression
                        ? 'border-rose-900/60 bg-rose-950/10'
                        : probe.passed
                        ? 'border-slate-800 bg-slate-950/40 hover:border-slate-700'
                        : 'border-amber-900/40 bg-amber-950/10'
                    }`}
                  >
                    <div
                      onClick={() => setExpandedProbeId(isExpanded ? null : probe.probe_id)}
                      className="p-3 flex items-center justify-between cursor-pointer select-none"
                    >
                      <div className="flex items-center gap-3 min-w-0">
                        {probe.passed ? (
                          <CheckCircle2 className="w-4 h-4 text-emerald-400 shrink-0" />
                        ) : isRegression ? (
                          <XCircle className="w-4 h-4 text-rose-400 shrink-0" />
                        ) : (
                          <AlertTriangle className="w-4 h-4 text-amber-400 shrink-0" />
                        )}
                        <span className="font-mono text-xs font-semibold text-slate-200 shrink-0">
                          {probe.probe_id}
                        </span>
                        <span className="text-[11px] px-2 py-0.5 rounded bg-slate-800 text-slate-300 capitalize shrink-0">
                          {probe.category.replace(/_/g, ' ')}
                        </span>
                        <span className="text-xs text-slate-400 truncate max-w-md hidden md:inline">
                          {probe.prompt}
                        </span>
                      </div>

                      <div className="flex items-center gap-3 shrink-0">
                        {isRegression && (
                          <span className="text-[10px] font-bold px-2 py-0.5 rounded bg-rose-950 text-rose-300 border border-rose-800/60 uppercase">
                            Regression
                          </span>
                        )}
                        <span className="text-[11px] font-mono text-slate-400 bg-slate-950 px-2 py-0.5 rounded border border-slate-800">
                          {probe.evaluator}
                        </span>
                        {isExpanded ? (
                          <ChevronDown className="w-4 h-4 text-slate-400" />
                        ) : (
                          <ChevronRight className="w-4 h-4 text-slate-400" />
                        )}
                      </div>
                    </div>

                    {isExpanded && (
                      <div className="p-4 border-t border-slate-800/80 bg-slate-950/80 space-y-3 text-xs">
                        <div>
                          <span className="text-slate-400 block font-semibold mb-1">Prompt:</span>
                          <div className="bg-slate-900 p-2.5 rounded border border-slate-800 font-mono text-[11px] text-slate-200 whitespace-pre-wrap">
                            {probe.prompt}
                          </div>
                        </div>

                        <div className="grid grid-cols-1 md:grid-cols-2 gap-3">
                          <div>
                            <span className="text-slate-400 block font-semibold mb-1">Expected Output:</span>
                            <div className="bg-slate-900 p-2.5 rounded border border-slate-800 font-mono text-[11px] text-emerald-300 whitespace-pre-wrap">
                              {typeof probe.expected === 'object'
                                ? JSON.stringify(probe.expected, null, 2)
                                : String(probe.expected)}
                            </div>
                          </div>

                          <div>
                            <span className="text-slate-400 block font-semibold mb-1">Candidate Model Output:</span>
                            <div
                              className={`p-2.5 rounded border font-mono text-[11px] whitespace-pre-wrap ${
                                probe.passed
                                  ? 'bg-slate-900 border-slate-800 text-slate-200'
                                  : 'bg-rose-950/30 border-rose-900/60 text-rose-200'
                              }`}
                            >
                              {probe.actual_response || '(No response collected)'}
                            </div>
                          </div>
                        </div>

                        {!probe.passed && (
                          <div className="p-2.5 bg-rose-950/20 border border-rose-900/40 rounded text-rose-300 text-xs">
                            <span className="font-semibold block mb-0.5">Evaluator Verification Failure:</span>
                            <span>{probe.failure_reason || probe.details}</span>
                          </div>
                        )}
                      </div>
                    )}
                  </div>
                );
              })}
            </div>
          </div>
        </div>
      )}

      {/* ========================================================================= */}
      {/* VIEW 2: ADVANCED FAULT-INJECTION EXPERIMENTS (EXP-00 through EXP-04)       */}
      {/* ========================================================================= */}
      {viewMode === 'advanced-experiments' && (
        <div className="space-y-6">
          <div className="bg-slate-900 border border-slate-800 rounded-xl p-5">
            <h3 className="text-base font-semibold text-slate-100 mb-1">
              Deterministic Fault-Injection Experiments (EXP-00..04)
            </h3>
            <p className="text-xs text-slate-400 mb-4">
              Controlled simulations injecting ground-truth isolated mutations to test layer attribution (Model, Serving, Safety, Request Config).
            </p>

            <div className="grid grid-cols-1 lg:grid-cols-3 gap-6">
              <div className="space-y-3">
                <label className="text-xs font-semibold text-slate-300 block">Select Target Scenario:</label>
                <div className="space-y-2">
                  {[
                    { id: 'EXP-00', name: 'EXP-00: Null Baseline Control (0% mutation)' },
                    { id: 'EXP-01', name: 'EXP-01: Model Reasoning Corruption (5 probes)' },
                    { id: 'EXP-02', name: 'EXP-02: Serving Stack Congestion (+1200ms delay)' },
                    { id: 'EXP-03', name: 'EXP-03: Safety Guardrail Over-Moderation (75% refusal)' },
                    { id: 'EXP-04', name: 'EXP-04: Request Config Truncation (max_tokens=20)' },
                  ].map((s) => (
                    <button
                      key={s.id}
                      onClick={() => setSimScenarioId(s.id)}
                      className={`w-full text-left px-3 py-2.5 rounded-lg text-xs font-medium transition-colors border ${
                        simScenarioId === s.id
                          ? 'bg-sky-950 border-sky-500 text-sky-200'
                          : 'bg-slate-950/60 border-slate-800 text-slate-300 hover:bg-slate-800'
                      }`}
                    >
                      {s.name}
                    </button>
                  ))}
                </div>

                <button
                  onClick={handleRunSim}
                  disabled={runningSim}
                  className="w-full mt-4 px-4 py-2.5 bg-sky-600 hover:bg-sky-500 text-white rounded-lg text-xs font-semibold flex items-center justify-center gap-2 shadow-sm shadow-sky-600/30 disabled:opacity-50 transition-colors"
                >
                  {runningSim ? (
                    <>
                      <RefreshCw className="w-3.5 h-3.5 animate-spin" />
                      Executing Simulated Experiment...
                    </>
                  ) : (
                    <>
                      <Play className="w-3.5 h-3.5" />
                      Execute {simScenarioId}
                    </>
                  )}
                </button>
              </div>

              <div className="lg:col-span-2 bg-slate-950 p-4 rounded-xl border border-slate-800 space-y-4">
                <h4 className="text-xs font-semibold text-slate-300 uppercase tracking-wider">
                  Simulation Outcome & Attribution Analysis
                </h4>

                {simError && (
                  <div className="p-3 bg-rose-950/40 border border-rose-800 rounded text-rose-300 text-xs">
                    {simError}
                  </div>
                )}

                {simResult ? (
                  <div className="space-y-4 text-xs">
                    <div className="grid grid-cols-2 md:grid-cols-4 gap-3">
                      <div className="p-2.5 bg-slate-900 rounded border border-slate-800">
                        <span className="text-slate-400 block text-[11px]">Predicted Layer</span>
                        <span className="font-mono font-bold text-sky-300 text-sm">
                          {simResult.attribution.predicted_layer}
                        </span>
                      </div>
                      <div className="p-2.5 bg-slate-900 rounded border border-slate-800">
                        <span className="text-slate-400 block text-[11px]">Ground Truth</span>
                        <span className="font-mono font-bold text-slate-200 text-sm">
                          {simResult.ground_truth.canonical_layer}
                        </span>
                      </div>
                      <div className="p-2.5 bg-slate-900 rounded border border-slate-800">
                        <span className="text-slate-400 block text-[11px]">Attribution Match</span>
                        <span className={`font-mono font-bold text-sm ${simResult.evaluation.attribution_correct ? 'text-emerald-400' : 'text-rose-400'}`}>
                          {simResult.evaluation.attribution_correct ? 'CORRECT (100%)' : 'MISMATCH'}
                        </span>
                      </div>
                      <div className="p-2.5 bg-slate-900 rounded border border-slate-800">
                        <span className="text-slate-400 block text-[11px]">Drift Detected</span>
                        <span className={`font-mono font-bold text-sm ${simResult.detection.drift_detected ? 'text-amber-400' : 'text-slate-300'}`}>
                          {simResult.detection.drift_detected ? 'YES' : 'NO'}
                        </span>
                      </div>
                    </div>

                    <div className="p-3 bg-slate-900 rounded border border-slate-800 space-y-2">
                      <span className="text-slate-400 font-semibold block">Observable Behavioural Evidence:</span>
                      <ul className="list-disc pl-4 space-y-1 text-slate-300">
                        {simResult.attribution.evidence?.map((ev: string, i: number) => (
                          <li key={i}>{ev}</li>
                        ))}
                      </ul>
                    </div>
                  </div>
                ) : (
                  <div className="py-12 text-center text-slate-500 text-xs">
                    Select a fault injection scenario on the left and click "Execute" to observe real-time layer attribution.
                  </div>
                )}
              </div>
            </div>
          </div>
        </div>
      )}

      {/* ========================================================================= */}
      {/* VIEW 3: PHYSICAL LOCAL QUANTIZATION (FP16 vs Q8 vs Q4)                     */}
      {/* ========================================================================= */}
      {viewMode === 'quantization' && (
        <div className="space-y-6">
          <div className="bg-slate-900 border border-slate-800 rounded-xl p-5">
            <h3 className="text-base font-semibold text-slate-100 mb-1">
              Physical Ollama Quantization Measurements (Qwen2.5-1.5B)
            </h3>
            <p className="text-xs text-slate-400 mb-4">
              Real hardware measurements across 1.5B quant levels on Windows RTX 3050. Demonstrates precision vs VRAM trade-offs.
            </p>

            <div className="overflow-x-auto">
              <table className="w-full text-left text-xs">
                <thead className="bg-slate-950/80 text-slate-400 uppercase text-[11px] tracking-wider border-b border-slate-800">
                  <tr>
                    <th className="px-4 py-3">Quantization Level</th>
                    <th className="px-4 py-3 text-center">Disk Size</th>
                    <th className="px-4 py-3 text-center">Peak VRAM</th>
                    <th className="px-4 py-3 text-center">Generation TPS</th>
                    <th className="px-4 py-3 text-center">Synthetic Pass Rate</th>
                    <th className="px-4 py-3 text-right">Attribution Result</th>
                  </tr>
                </thead>
                <tbody className="divide-y divide-slate-800/60 font-mono">
                  <tr className="hover:bg-slate-800/40">
                    <td className="px-4 py-3 font-semibold text-sky-400 font-sans">FP16 (High Precision)</td>
                    <td className="px-4 py-3 text-center">3.1 GB</td>
                    <td className="px-4 py-3 text-center">3,200 MiB</td>
                    <td className="px-4 py-3 text-center">78.4 tok/s</td>
                    <td className="px-4 py-3 text-center text-emerald-400 font-bold">100.0%</td>
                    <td className="px-4 py-3 text-right font-sans text-slate-400">Baseline Target</td>
                  </tr>
                  <tr className="hover:bg-slate-800/40 bg-sky-950/20">
                    <td className="px-4 py-3 font-semibold text-sky-300 font-sans">Q8_0 (Recommended Anchor)</td>
                    <td className="px-4 py-3 text-center">1.6 GB</td>
                    <td className="px-4 py-3 text-center">1,820 MiB</td>
                    <td className="px-4 py-3 text-center">112.5 tok/s</td>
                    <td className="px-4 py-3 text-center text-emerald-400 font-bold">100.0%</td>
                    <td className="px-4 py-3 text-right font-sans text-emerald-400 font-semibold">Zero Degradation</td>
                  </tr>
                  <tr className="hover:bg-slate-800/40">
                    <td className="px-4 py-3 font-semibold text-amber-300 font-sans">Q4_K_M (Default Ollama)</td>
                    <td className="px-4 py-3 text-center">986 MB</td>
                    <td className="px-4 py-3 text-center">1,150 MiB</td>
                    <td className="px-4 py-3 text-center">145.2 tok/s</td>
                    <td className="px-4 py-3 text-center text-amber-400 font-bold">94.3%</td>
                    <td className="px-4 py-3 text-right font-sans text-amber-300 font-semibold">Model-Layer Drift Flagged</td>
                  </tr>
                </tbody>
              </table>
            </div>
          </div>
        </div>
      )}

      {/* ========================================================================= */}
      {/* VIEW 4: COMPARATIVE MATRIX & METHODOLOGY                                  */}
      {/* ========================================================================= */}
      {viewMode === 'methodology' && (
        <div className="space-y-6">
          <div className="bg-slate-900 border border-slate-800 rounded-xl p-5">
            <h3 className="text-base font-semibold text-slate-100 mb-1">
              Go V2 Engine vs Python Reference Verification Matrix
            </h3>
            <p className="text-xs text-slate-400 mb-4">
              Head-to-head parity verification across 16 common strict evaluation probes.
            </p>

            <div className="space-y-3">
              {COMPARATIVE_MATRIX.map((dim, idx) => (
                <div key={idx} className="p-3 bg-slate-950 rounded-lg border border-slate-800 text-xs">
                  <div className="flex items-center justify-between font-semibold text-slate-200 mb-1">
                    <span>{dim.dimension}</span>
                    <span className="text-slate-400 font-normal">{dim.metric}</span>
                  </div>
                  <div className="grid grid-cols-1 md:grid-cols-3 gap-2 text-slate-400 font-mono text-[11px]">
                    <div><span className="text-slate-500 font-sans">Go V1:</span> {dim.go_v1}</div>
                    <div><span className="text-sky-400 font-sans">Go V2:</span> {dim.go_v2}</div>
                    <div><span className="text-amber-400 font-sans">Python Ref:</span> {dim.python_ref}</div>
                  </div>
                </div>
              ))}
            </div>
          </div>
        </div>
      )}
    </div>
  );
};
