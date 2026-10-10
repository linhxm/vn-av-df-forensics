import { useEffect, useRef, useState } from "react";
import { createRoot } from "react-dom/client";
import "./pipeline.css";

type Branch = "sync" | "artifact";
type Result = {
  video_score: number | null; coverage?: number; seed?: number; method?: string;
  intervals: { start_sec: number; end_sec: number; score: number }[];
  scores?: number[]; times_s?: number[]; valid?: boolean[];
  thresholds?: { video: number | null; temporal: number | null };
  // P2: điểm từng nhánh bằng chứng; chưa có ngưỡng riêng nên không có khoảng thời gian.
  evidence?: Partial<Record<Branch, number | null>>;
  evidence_scores?: Partial<Record<Branch, number[]>>;
};
type ResearchRun = { architecture: string; seed: number; video: { roc_auc: number | null; pr_auc: number | null; f1: number | null } };
type Job = { id: string; status: string; result?: Result; error?: string };
const time = (seconds: number) => `${Math.floor(seconds / 60).toString().padStart(2, "0")}:${(seconds % 60).toFixed(1).padStart(4, "0")}`;
const score = (x: number | null) => x == null ? "Chưa đủ dữ liệu để chấm điểm" : x.toFixed(3);
const BRANCHES: { key: Branch; label: string; color: string; note: string }[] = [
  { key: "sync", label: "Nhánh sync (bất nhất quán tiếng-miệng)", color: "#b35c00", note: "Học từ real, sham và real bị dịch lệch tiếng; chưa từng thấy video fake." },
  { key: "artifact", label: "Nhánh artifact (dấu vết vùng miệng)", color: "#2f7d4a", note: "DINOv2 trên crop miệng, học cùng nhãn AI." },
];

// Mỗi cột là một ô 0,2 s; ô thiếu quan sát để trống, bấm vào cột để tua video.
function Timeline({ label, values, valid, times, color, threshold, onSeek }: {
  label: string; values: number[]; valid?: boolean[]; times?: number[]; color: string;
  threshold?: number | null; onSeek: (seconds: number) => void;
}) {
  const width = 800 / values.length;
  return <svg viewBox="0 0 800 110" role="img" aria-label={label} style={{ width: "100%", background: "#eef2f7" }}>
    {values.map((value, i) => valid?.[i] !== false && <rect key={i} x={i * width} y={100 - value * 100} width={width} height={value * 100} fill={color}
      onClick={() => { if (times) onSeek(times[i]); }} />)}
    {threshold != null && <line x1={0} x2={800} y1={100 - threshold * 100} y2={100 - threshold * 100} stroke="#555" strokeDasharray="6 4" />}
  </svg>;
}

function App() {
  const [file, setFile] = useState<File | null>(null);
  const [job, setJob] = useState<Job | null>(null);
  const [error, setError] = useState("");
  const [uploading, setUploading] = useState(false);
  const [ready, setReady] = useState(false);
  const [research, setResearch] = useState<{ runs: ResearchRun[] | null } | null>(null);
  const video = useRef<HTMLVideoElement>(null);
  const busy = uploading || job?.status === "queued" || job?.status === "processing";
  useEffect(() => {
    fetch("/api/health").then(r => r.json()).then(x => setReady(x.ready)).catch(() => setReady(false));
    fetch("/api/research").then(r => r.json()).then(setResearch).catch(() => setResearch(null));
  }, []);
  useEffect(() => {
    if (!job || !["queued", "processing"].includes(job.status)) return;
    let active = true;
    const timer = setTimeout(async () => {
      try {
        const r = await fetch(`/api/jobs/${job.id}`);
        if (!r.ok) throw Error(await r.text());
        const next = await r.json();
        if (active) setJob(next);
      } catch (e) { if (active) { setError(String(e)); setJob({ ...job, status: "failed" }); } }
    }, 1000);
    return () => { active = false; clearTimeout(timer); };
  }, [job]);
  async function submit() {
    if (!file) return;
    setUploading(true); setJob(null); setError("");
    try {
      const body = new FormData(); body.append("video", file);
      const r = await fetch("/api/jobs", { method: "POST", body });
      if (!r.ok) throw Error(await r.text());
      setJob(await r.json());
    } catch (e) { setError(String(e)); } finally { setUploading(false); }
  }
  const result = job?.result;
  // Kết quả test từng detector của run đang chọn (evaluation.json); so sánh đầy đủ ở compare.ipynb.
  const researchRuns = research?.runs ?? null;
  const seek = (seconds: number) => { if (video.current) video.current.currentTime = seconds; };
  return <main>
    <h1>VN-AV-DF Forensics</h1>
    <p>Phát hiện deepfake và định vị khoảng nghi ngờ. Video một người nói, có audio và thấy rõ mặt; tối đa 60 giây trong cấu hình hiện tại.</p>
    <input aria-label="Video" type="file" accept="video/*" disabled={busy} onChange={e => setFile(e.target.files?.[0] ?? null)} />
    <button disabled={!file || busy || !ready} onClick={submit}>{busy ? "Đang xử lý…" : "Phân tích"}</button>
    {!ready && <p>Chưa có checkpoint đã train. Thu thập, duyệt dữ liệu và chạy training trước khi phân tích.</p>}
    <p role="status">{job?.status === "queued" ? "Đang chờ" : job?.status === "processing" ? "Đang phân tích video" : ""}</p>
    {(error || job?.error) && <p role="alert">{error || job?.error}</p>}
    {job && result && <section>
      <h2>Điểm nghi ngờ toàn video: {score(result.video_score)}</h2>
      <p>Điểm từ 0 đến 1, chưa phải xác suất giả mạo đã hiệu chuẩn. Không có khoảng vượt ngưỡng không xác nhận video là thật.</p>
      <video ref={video} src={`/api/jobs/${job.id}/media`} controls />
      {result.scores && <>
        <h3>Điểm nghi ngờ AI theo thời gian</h3>
        <Timeline label="Điểm nghi ngờ theo thời gian" values={result.scores} valid={result.valid} times={result.times_s}
          color="#3256a8" threshold={result.thresholds?.temporal} onSeek={seek} />
        {result.thresholds?.temporal != null && <small>Đường nét đứt: ngưỡng định vị chọn trên validation.</small>}
      </>}
      {result.coverage != null && <p>Tỷ lệ thời gian quan sát được: {(result.coverage * 100).toFixed(1)}%. Khoảng trống trên đồ thị là thiếu quan sát.</p>}
      {result.evidence && <>
        <h2>Bằng chứng theo từng nhánh</h2>
        <p className="warning">Điểm từng nhánh là bằng chứng phụ để giải thích, chưa có ngưỡng riêng và không thay điểm AI.
          Nhánh sync cao nghĩa là tiếng và miệng lệch nhau; điều này cũng xảy ra với video lồng tiếng thông thường không dùng AI.</p>
        <table><thead><tr><th>Nhánh</th><th>Điểm toàn video</th><th>Cách học</th></tr></thead>
          <tbody>{BRANCHES.filter(b => b.key in result.evidence!).map(b => <tr key={b.key}>
            <td>{b.label}</td><td>{score(result.evidence![b.key] ?? null)}</td><td>{b.note}</td>
          </tr>)}</tbody></table>
        {BRANCHES.map(b => result.evidence_scores?.[b.key] && <div key={b.key}>
          <h3>{b.label}</h3>
          <Timeline label={b.label} values={result.evidence_scores[b.key]!} valid={result.valid} times={result.times_s} color={b.color} onSeek={seek} />
        </div>)}
      </>}
      <h2>Các khoảng nghi ngờ</h2>
      <table><thead><tr><th>Bắt đầu</th><th>Kết thúc</th><th>Điểm</th></tr></thead>
        <tbody>{result.intervals.map((span, i) => <tr key={i}>
          <td><button onClick={() => { if (video.current) video.current.currentTime = span.start_sec; }}>{time(span.start_sec)}</button></td>
          <td>{time(span.end_sec)}</td><td>{score(span.score)}</td>
        </tr>)}</tbody></table>
      {!result.intervals.length && <p>Không có khoảng được báo trong lần phân tích này.</p>}
      <p><a href={`/api/jobs/${job.id}/report`} download>Tải kết quả JSON</a></p>
    </section>}
    <details><summary>Kết quả test từng detector (run đang chọn)</summary>{researchRuns ? <table>
      <thead><tr><th>Phương pháp</th><th>Seed</th><th>ROC-AUC</th><th>PR-AUC</th><th>F1</th></tr></thead>
      <tbody>{researchRuns.map(r => <tr key={`${r.architecture}-${r.seed}`}><td>{r.architecture}</td><td>{r.seed}</td><td>{score(r.video.roc_auc)}</td><td>{score(r.video.pr_auc)}</td><td>{score(r.video.f1)}</td></tr>)}</tbody>
    </table> : <p>Chưa có kết quả test của bộ dữ liệu mới.</p>}</details>
  </main>;
}
createRoot(document.getElementById("root")!).render(<App />);
