import { useEffect, useRef, useState } from "react";
import {
  Activity, ArrowDownRight, ArrowUpRight, BarChart3, Bell,
  CheckCircle2, ChevronRight, CircleHelp, Clock3, Cpu,
  Database, Download, FileImage, FolderOpen, Gauge,
  ImagePlus, Info, Layers3, LayoutDashboard, Menu,
  Monitor, Moon, Play, ScanEye, Settings, ShieldCheck,
  Sparkles, Upload, X, Zap
} from "lucide-react";
import {
  Area, AreaChart, CartesianGrid, Legend, ResponsiveContainer,
  Tooltip, XAxis, YAxis, BarChart, Bar
} from "recharts";
import "./App.css";

const navItems = [
  { id: "overview", label: "Overview", icon: LayoutDashboard, group: "WORKSPACE" },
  { id: "enhance", label: "Image Enhancement", icon: ScanEye, group: "WORKSPACE" },
  { id: "performance", label: "Model Performance", icon: BarChart3, group: "ANALYTICS" },
  { id: "dataset", label: "Dataset Explorer", icon: Database, group: "ANALYTICS" },
  { id: "experiments", label: "Experiments", icon: Layers3, group: "ANALYTICS" },
  { id: "about", label: "About Project", icon: Info, group: "PROJECT" },
];

const metricData = [
  { name: "Bicubic", psnr: 26.0917, ssim: 0.725639 },
  { name: "SRCNN", psnr: 26.3103, ssim: 0.742897 },
];

const epochData = [
  { epoch: "Baseline", psnr: 26.0917 },
  { epoch: "SRCNN", psnr: 26.3103 },
];

const experiments = [
  { name: "SRCNN — DIV2K full-image evaluation", date: "13 Nov 2026", type: "Evaluation", status: "Completed", detail: "100 validation images" },
  { name: "SRCNN — 5 epoch training", date: "12 Nov 2026", type: "Training", status: "Completed", detail: "Best checkpoint evaluation" },
  { name: "DIV2K degradation pipeline", date: "12 Nov 2026", type: "Data Pipeline", status: "Completed", detail: "800 train · 100 validation" },
  { name: "SRCNN patch dataset tests", date: "12 Nov 2026", type: "Testing", status: "Passed", detail: "31 project tests passed" },
];

function StatCard({ title, value, unit, caption, icon: Icon, trend, positive = true }) {
  return (
    <div className="stat-card">
      <div className="stat-top">
        <span>{title}</span>
        <span className="stat-icon"><Icon size={18} /></span>
      </div>
      <div className="stat-value">{value}<small>{unit}</small></div>
      <div className="stat-bottom">
        <span className={positive ? "trend positive" : "trend neutral"}>
          {trend && (positive ? <ArrowUpRight size={14} /> : <ArrowDownRight size={14} />)}
          {trend}
        </span>
        <span className="stat-caption">{caption}</span>
      </div>
    </div>
  );
}

function SectionHeading({ eyebrow, title, description, action }) {
  return (
    <div className="section-heading">
      <div>
        <div className="eyebrow">{eyebrow}</div>
        <h1>{title}</h1>
        <p>{description}</p>
      </div>
      {action}
    </div>
  );
}

function Overview({ navigate }) {
  return (
    <>
      <SectionHeading
        eyebrow="PROJECT WORKSPACE / OVERVIEW"
        title="Surveillance SR Studio"
        description="An experimental image super-resolution framework for enhancing low-quality surveillance imagery."
        action={<span className="date-chip"><Clock3 size={14} /> Project snapshot · 13 Nov 2026</span>}
      />

      <div className="hero-panel">
        <div className="hero-copy">
          <div className="live-label"><span className="live-dot" /> RESEARCH PROJECT</div>
          <h2>Recover more visual detail.<br /><span>Evaluate with confidence.</span></h2>
          <p>Explore the SRCNN pipeline, inspect benchmark performance, and prepare low-resolution images for model inference.</p>
          <button className="primary-btn" onClick={() => navigate("enhance")}>
            <ImagePlus size={17} /> Open enhancement studio <ChevronRight size={16} />
          </button>
        </div>
        <div className="hero-art" aria-hidden="true">
          <div className="art-ring ring-one" />
          <div className="art-ring ring-two" />
          <div className="art-core"><ScanEye size={62} strokeWidth={1.1} /></div>
          <div className="art-tag tag-top"><Activity size={13} /> 4× SR PIPELINE</div>
          <div className="art-tag tag-bottom"><Cpu size={13} /> SRCNN MODEL</div>
        </div>
      </div>

      <div className="stats-grid">
        <StatCard title="Validation images" value="100" caption="DIV2K validation set" icon={FileImage} trend="Verified" />
        <StatCard title="SRCNN PSNR" value="26.31" unit=" dB" caption="Full-image evaluation" icon={Gauge} trend="+0.22 dB" />
        <StatCard title="SRCNN SSIM" value="0.743" caption="Structural similarity" icon={ScanEye} trend="+0.0173" />
        <StatCard title="Avg. inference" value="0.765" unit=" s" caption="Per image · CPU" icon={Zap} trend="Measured" />
      </div>

      <div className="content-grid">
        <div className="panel performance-panel">
          <div className="panel-heading">
            <div><h3>Benchmark performance</h3><p>Full-image DIV2K validation · 100 images</p></div>
            <button className="text-btn" onClick={() => navigate("performance")}>Details <ChevronRight size={14} /></button>
          </div>
          <div className="chart-wrap">
            <ResponsiveContainer width="100%" height="100%">
              <BarChart data={metricData} barSize={42} margin={{ top: 10, right: 12, left: -18, bottom: 0 }}>
                <CartesianGrid strokeDasharray="3 3" vertical={false} stroke="#253044" />
                <XAxis dataKey="name" tick={{ fill: "#98a5ba", fontSize: 12 }} axisLine={false} tickLine={false} />
                <YAxis domain={[25.8, 26.5]} tick={{ fill: "#77869e", fontSize: 11 }} axisLine={false} tickLine={false} />
                <Tooltip contentStyle={{ background: "#111c2d", border: "1px solid #2b3b54", borderRadius: 10, color: "#eef4ff" }} />
                <Bar dataKey="psnr" name="PSNR (dB)" fill="#4e9bff" radius={[6, 6, 0, 0]} />
              </BarChart>
            </ResponsiveContainer>
          </div>
          <div className="chart-footnote"><Info size={13} /> Benchmark results are from the recorded DIV2K full-image evaluation, not uploaded images.</div>
        </div>

        <div className="panel model-panel">
          <div className="panel-heading"><div><h3>Model status</h3><p>Current development snapshot</p></div><span className="status-pill status-ready">Checkpoint reported</span></div>
          <div className="model-visual"><div className="model-symbol"><Cpu size={30} /></div><div><strong>SRCNN</strong><span>Super-Resolution CNN</span></div></div>
          <div className="model-details">
            <div><span>Architecture</span><strong>9×9 · 1×1 · 5×5</strong></div>
            <div><span>Parameters</span><strong>20,099</strong></div>
            <div><span>Scale factor</span><strong>4×</strong></div>
            <div><span>Runtime device</span><strong>CPU</strong></div>
          </div>
          <div className="notice"><Info size={15} /><span>FastAPI inference connection is not configured yet. Uploaded images are currently preview-only.</span></div>
        </div>
      </div>

      <div className="panel activity-panel">
        <div className="panel-heading"><div><h3>Recent project activity</h3><p>Recorded development milestones</p></div><button className="text-btn" onClick={() => navigate("experiments")}>View all <ChevronRight size={14} /></button></div>
        <div className="activity-list">
          {experiments.slice(0, 3).map((item, i) => (
            <div className="activity-row" key={item.name}>
              <div className={`activity-icon activity-${i}`}><CheckCircle2 size={16} /></div>
              <div className="activity-main"><strong>{item.name}</strong><span>{item.detail}</span></div>
              <span className="activity-date">{item.date}</span>
              <span className="status-pill status-ready">{item.status}</span>
            </div>
          ))}
        </div>
      </div>
    </>
  );
}

function Enhancement({ file, setFile, preview, setPreview }) {
  const inputRef = useRef(null);
  const [message, setMessage] = useState("");
  const [resizedUrl, setResizedUrl] = useState("");
  const [loading, setLoading] = useState(false);
  const [metadata, setMetadata] = useState(null);

  useEffect(() => {
    return () => {
      if (resizedUrl) URL.revokeObjectURL(resizedUrl);
    };
  }, [resizedUrl]);

  function handleFile(selected) {
    if (!selected) return;

    const allowed = [
      "image/png",
      "image/jpeg",
      "image/webp",
      "image/bmp",
      "image/tiff",
    ];

    if (!allowed.includes(selected.type)) {
      setMessage("Unsupported image. Choose PNG, JPEG, WebP, BMP or TIFF.");
      return;
    }

    if (selected.size > 15 * 1024 * 1024) {
      setMessage("Image exceeds the 15 MB upload limit.");
      return;
    }

    setFile(selected);
    setPreview(URL.createObjectURL(selected));
    setResizedUrl("");
    setMetadata(null);
    setMessage("Image ready. Click Enhance with SRCNN to run the trained model.");
  }

  async function runEnhancement() {
    if (!file || loading) return;

    setLoading(true);
    setMessage("Uploading image and running SRCNN inference...");
    setResizedUrl("");
    setMetadata(null);

    try {
      const formData = new FormData();
      formData.append("file", file);

      const response = await fetch("/api/enhance", {
        method: "POST",
        body: formData,
      });

      if (!response.ok) {
        let detail = `API request failed (${response.status})`;
        try {
          const error = await response.json();
          detail = error.detail || detail;
        } catch {
          // Keep the HTTP status message if the response is not JSON.
        }
        throw new Error(detail);
      }

      const blob = await response.blob();
      const outputUrl = URL.createObjectURL(blob);

      setResizedUrl(outputUrl);
      setMetadata({
        seconds: response.headers.get("X-Inference-Seconds"),
        inputWidth: response.headers.get("X-Input-Width"),
        inputHeight: response.headers.get("X-Input-Height"),
        outputWidth: response.headers.get("X-Output-Width"),
        outputHeight: response.headers.get("X-Output-Height"),
        scale: response.headers.get("X-Scale-Factor"),
        model: response.headers.get("X-Model-Name"),
      });

      setMessage("SRCNN enhancement completed successfully.");
    } catch (error) {
      setMessage(
        `${error.message}. Check that FastAPI is running on port 8000.`
      );
    } finally {
      setLoading(false);
    }
  }

  function clearImage() {
    setFile(null);
    setPreview("");
    setResizedUrl("");
    setMetadata(null);
    setMessage("");
    if (inputRef.current) inputRef.current.value = "";
  }

  return (
    <>
      <SectionHeading
        eyebrow="WORKSPACE / IMAGE PROCESSING"
        title="Image Enhancement"
        description="Upload an image and enhance it using your trained SRCNN model through the FastAPI backend."
      />

      <div className="enhance-layout">
        <div className="panel upload-panel">
          <div className="panel-heading">
            <div>
              <h3>Input image</h3>
              <p>PNG, JPG, JPEG, WEBP, BMP, TIFF · Max 15 MB</p>
            </div>
            <span className="status-pill status-ready">SRCNN connected</span>
          </div>

          {!file ? (
            <button
              className="drop-zone"
              onClick={() => inputRef.current?.click()}
              onDragOver={(event) => event.preventDefault()}
              onDrop={(event) => {
                event.preventDefault();
                handleFile(event.dataTransfer.files?.[0]);
              }}
            >
              <div className="upload-icon"><Upload size={25} /></div>
              <strong>Drop your image here</strong>
              <span>or click to browse files</span>
              <small>Image is sent to your local FastAPI backend for inference.</small>
            </button>
          ) : (
            <div className="image-preview-box">
              <div className="preview-toolbar">
                <span><FileImage size={14} /> {file.name}</span>
                <button
                  className="icon-btn"
                  onClick={clearImage}
                  title="Remove image"
                  disabled={loading}
                >
                  <X size={16} />
                </button>
              </div>
              <img
                src={preview}
                alt="Uploaded original image"
                className="uploaded-image"
              />
              <div className="image-meta">
                <span>{(file.size / 1024).toFixed(1)} KB</span>
                <span>{file.type}</span>
              </div>
            </div>
          )}

          <input
            ref={inputRef}
            type="file"
            accept="image/png,image/jpeg,image/webp,image/bmp,image/tiff"
            hidden
            onChange={(event) => handleFile(event.target.files?.[0])}
          />

          <div className="settings-block">
            <h4>Inference settings</h4>
            <div className="model-details">
              <div><span>Model</span><strong>SRCNN</strong></div>
              <div><span>Scale factor</span><strong>4×</strong></div>
              <div><span>Device</span><strong>CPU</strong></div>
              <div><span>Architecture</span><strong>9×9 · 1×1 · 5×5</strong></div>
            </div>

            <div className="notice">
              <Info size={15} />
              <span>Actual PyTorch SRCNN inference is performed by FastAPI. This is not browser interpolation.</span>
            </div>

            <button
              className="primary-btn full-btn"
              disabled={!file || loading}
              onClick={runEnhancement}
            >
              <Sparkles size={16} />
              {loading ? "Running SRCNN..." : "Enhance with SRCNN"}
            </button>

            {message && <p className="inline-message">{message}</p>}
          </div>
        </div>

        <div className="panel output-panel">
          <div className="panel-heading">
            <div>
              <h3>Enhanced output</h3>
              <p>Trained SRCNN model output</p>
            </div>
            <span className={`status-pill ${resizedUrl ? "status-ready" : "status-neutral"}`}>
              {resizedUrl ? "AI enhanced" : "Waiting"}
            </span>
          </div>

          {resizedUrl ? (
            <>
              <div className="output-image-wrap">
                <img src={resizedUrl} alt="SRCNN enhanced image" />
              </div>

              {metadata && (
                <div className="model-details">
                  <div><span>Model</span><strong>{metadata.model}</strong></div>
                  <div><span>Inference time</span><strong>{metadata.seconds} s</strong></div>
                  <div>
                    <span>Input dimensions</span>
                    <strong>{metadata.inputWidth} × {metadata.inputHeight}</strong>
                  </div>
                  <div>
                    <span>Output dimensions</span>
                    <strong>{metadata.outputWidth} × {metadata.outputHeight}</strong>
                  </div>
                  <div><span>Scale</span><strong>{metadata.scale}×</strong></div>
                </div>
              )}

              <a
                className="download-btn"
                href={resizedUrl}
                download={`srcnn-enhanced-${file?.name?.replace(/\.[^.]+$/, "") || "image"}.png`}
              >
                <Download size={16} /> Download enhanced PNG
              </a>
            </>
          ) : (
            <div className="empty-output">
              <div className="empty-output-icon"><ImagePlus size={28} /></div>
              <strong>Your enhanced image will appear here</strong>
              <span>Upload an image and run SRCNN inference.</span>
            </div>
          )}

          <div className="pipeline-steps">
            <div className={`pipeline-step ${file ? "done" : "current"}`}>
              <span>1</span>
              <div><strong>Image upload</strong><small>React frontend</small></div>
              {file ? <CheckCircle2 size={16} /> : <Clock3 size={16} />}
            </div>
            <div className={`pipeline-step ${resizedUrl ? "done" : loading ? "current" : "locked"}`}>
              <span>2</span>
              <div><strong>FastAPI processing</strong><small>4× bicubic preprocessing</small></div>
              {resizedUrl ? <CheckCircle2 size={16} /> : <Clock3 size={16} />}
            </div>
            <div className={`pipeline-step ${resizedUrl ? "done" : loading ? "current" : "locked"}`}>
              <span>3</span>
              <div><strong>SRCNN inference</strong><small>PyTorch · CPU · Epoch 5 checkpoint</small></div>
              {resizedUrl ? <CheckCircle2 size={16} /> : <ShieldCheck size={16} />}
            </div>
          </div>
        </div>
      </div>

      <div className="warning-box">
        <ShieldCheck size={18} />
        <div>
          <strong>Research and evidence integrity</strong>
          <p>Super-resolution can generate or alter visual details. Enhanced images must be labelled as AI-reconstructed and must not be treated as proof of a person's identity, a number plate, or other forensic fact. Preserve the original image.</p>
        </div>
      </div>
    </>
  );
}
function Performance() {
  return (
    <>
      <SectionHeading eyebrow="ANALYTICS / BENCHMARKS" title="Model Performance" description="Recorded full-image evaluation results on the DIV2K validation dataset." />
      <div className="stats-grid">
        <StatCard title="Bicubic PSNR" value="26.092" unit=" dB" caption="Baseline" icon={Gauge} />
        <StatCard title="SRCNN PSNR" value="26.310" unit=" dB" caption="Model output" icon={Activity} trend="+0.219 dB" />
        <StatCard title="Bicubic SSIM" value="0.7256" caption="Baseline" icon={ScanEye} />
        <StatCard title="SRCNN SSIM" value="0.7429" caption="Model output" icon={ShieldCheck} trend="+0.0173" />
      </div>
      <div className="content-grid">
        <div className="panel">
          <div className="panel-heading"><div><h3>PSNR comparison</h3><p>Higher is better · dB</p></div></div>
          <div className="chart-wrap tall-chart"><ResponsiveContainer width="100%" height="100%"><BarChart data={metricData} margin={{ top: 12, right: 15, left: -15, bottom: 5 }}><CartesianGrid stroke="#253044" strokeDasharray="3 3" vertical={false} /><XAxis dataKey="name" tick={{ fill: "#98a5ba", fontSize: 12 }} axisLine={false} tickLine={false} /><YAxis domain={[25.8, 26.5]} tick={{ fill: "#98a5ba", fontSize: 11 }} axisLine={false} tickLine={false} /><Tooltip contentStyle={{ background: "#111c2d", border: "1px solid #2b3b54", borderRadius: 10 }} /><Bar dataKey="psnr" fill="#4e9bff" radius={[7, 7, 0, 0]} /></BarChart></ResponsiveContainer></div>
        </div>
        <div className="panel">
          <div className="panel-heading"><div><h3>SSIM comparison</h3><p>Closer to 1.0 is better</p></div></div>
          <div className="chart-wrap tall-chart"><ResponsiveContainer width="100%" height="100%"><BarChart data={metricData} margin={{ top: 12, right: 15, left: -15, bottom: 5 }}><CartesianGrid stroke="#253044" strokeDasharray="3 3" vertical={false} /><XAxis dataKey="name" tick={{ fill: "#98a5ba", fontSize: 12 }} axisLine={false} tickLine={false} /><YAxis domain={[0.68, 0.78]} tick={{ fill: "#98a5ba", fontSize: 11 }} axisLine={false} tickLine={false} /><Tooltip contentStyle={{ background: "#111c2d", border: "1px solid #2b3b54", borderRadius: 10 }} /><Bar dataKey="ssim" fill="#29c7a0" radius={[7, 7, 0, 0]} /></BarChart></ResponsiveContainer></div>
        </div>
      </div>
      <div className="panel table-panel">
        <div className="panel-heading"><div><h3>Evaluation summary</h3><p>100 complete DIV2K validation images</p></div><span className="status-pill status-ready">Evaluation completed</span></div>
        <div className="table-scroll"><table><thead><tr><th>Metric</th><th>Bicubic baseline</th><th>SRCNN</th><th>Improvement</th></tr></thead><tbody>
          <tr><td>PSNR</td><td>26.091706 dB</td><td>26.310339 dB</td><td className="positive-text">+0.218634 dB</td></tr>
          <tr><td>SSIM</td><td>0.725639</td><td>0.742897</td><td className="positive-text">+0.017258</td></tr>
          <tr><td>Average inference time</td><td>—</td><td>0.765 sec/image</td><td>CPU</td></tr>
          <tr><td>Validation count</td><td>100</td><td>100</td><td>Full images</td></tr>
        </tbody></table></div>
        <div className="chart-footnote"><Info size={13} /> Average improvement does not mean every validation image improved. These results are from DIV2K, not CCTV-specific validation.</div>
      </div>
    </>
  );
}

function Dataset() {
  return (
    <>
      <SectionHeading eyebrow="ANALYTICS / DATA" title="Dataset Explorer" description="Overview of the DIV2K dataset and the synthetic degradation pipeline used in this project." />
      <div className="stats-grid">
        <StatCard title="Training images" value="800" caption="DIV2K HR images" icon={Database} />
        <StatCard title="Validation images" value="100" caption="DIV2K HR images" icon={FileImage} />
        <StatCard title="Scale factor" value="4×" caption="Low-resolution generation" icon={Layers3} />
        <StatCard title="Patch size" value="128" unit=" px" caption="HR training patches" icon={ScanEye} />
      </div>
      <div className="panel">
        <div className="panel-heading"><div><h3>Degradation pipeline</h3><p>Synthetic low-resolution input generation</p></div><span className="status-pill status-ready">Implemented</span></div>
        <div className="degradation-flow">
          {[
            ["01", "High-resolution image", "DIV2K source image", "image"],
            ["02", "Gaussian blur", "Blur degradation", "blur"],
            ["03", "Noise + JPEG", "Compression artifacts", "noise"],
            ["04", "Downsampling", "4× lower resolution", "down"],
            ["05", "LR input", "Model input preparation", "output"],
          ].map((step, i) => (
            <div className="degradation-item" key={step[0]}>
              <div className="degradation-number">{step[0]}</div>
              <div className="degradation-symbol">{i === 0 ? <FileImage /> : i === 1 ? <ScanEye /> : i === 2 ? <Activity /> : i === 3 ? <Layers3 /> : <Database />}</div>
              <strong>{step[1]}</strong><span>{step[2]}</span>
              {i < 4 && <ChevronRight className="flow-arrow" size={18} />}
            </div>
          ))}
        </div>
      </div>
      <div className="content-grid">
        <div className="panel"><div className="panel-heading"><div><h3>Dataset split</h3><p>DIV2K image counts</p></div></div><div className="split-bar"><span className="train-bar" /><span className="val-bar" /></div><div className="legend-row"><span><i className="legend-dot train-dot" /> Training <strong>800 (88.9%)</strong></span><span><i className="legend-dot val-dot" /> Validation <strong>100 (11.1%)</strong></span></div></div>
        <div className="panel"><div className="panel-heading"><div><h3>Data integrity</h3><p>Pipeline validation status</p></div></div><div className="integrity-item"><CheckCircle2 size={17} /><span>800 training pairs generated</span><strong>Passed</strong></div><div className="integrity-item"><CheckCircle2 size={17} /><span>100 validation pairs generated</span><strong>Passed</strong></div><div className="integrity-item"><CheckCircle2 size={17} /><span>0 validation errors reported</span><strong>Passed</strong></div></div>
      </div>
      <div className="notice"><Info size={16} /><span>Current benchmark dataset is DIV2K natural imagery with synthetic degradation. CCTV domain generalization still needs separate evaluation.</span></div>
    </>
  );
}

function Experiments() {
  return (
    <>
      <SectionHeading eyebrow="ANALYTICS / RESEARCH LOG" title="Experiments & Milestones" description="Recorded development and evaluation activity for the super-resolution framework." />
      <div className="panel table-panel">
        <div className="panel-heading"><div><h3>Experiment history</h3><p>Project records available in this dashboard snapshot</p></div><span className="status-pill status-ready">4 records</span></div>
        <div className="table-scroll"><table><thead><tr><th>Experiment</th><th>Type</th><th>Date</th><th>Details</th><th>Status</th></tr></thead><tbody>{experiments.map((e) => <tr key={e.name}><td className="experiment-name">{e.name}</td><td>{e.type}</td><td>{e.date}</td><td>{e.detail}</td><td><span className="status-pill status-ready">{e.status}</span></td></tr>)}</tbody></table></div>
      </div>
      <div className="panel">
        <div className="panel-heading"><div><h3>Development roadmap</h3><p>Upcoming engineering milestones</p></div></div>
        <div className="roadmap">
          <div className="roadmap-item complete"><span className="roadmap-marker"><CheckCircle2 size={16} /></span><div><strong>Data pipeline and SRCNN baseline</strong><p>Dataset preparation, model implementation, and tests.</p><small>COMPLETED</small></div></div>
          <div className="roadmap-item complete"><span className="roadmap-marker"><CheckCircle2 size={16} /></span><div><strong>Benchmark evaluation</strong><p>Full-image DIV2K evaluation and metrics.</p><small>COMPLETED</small></div></div>
          <div className="roadmap-item current"><span className="roadmap-marker"><Monitor size={16} /></span><div><strong>Frontend showcase</strong><p>Dashboard, upload workspace, and performance views.</p><small>IN PROGRESS</small></div></div>
          <div className="roadmap-item"><span className="roadmap-marker"><Cpu size={16} /></span><div><strong>FastAPI + real SRCNN inference</strong><p>Connect the trained checkpoint to the browser application.</p><small>UP NEXT</small></div></div>
          <div className="roadmap-item"><span className="roadmap-marker"><BarChart3 size={16} /></span><div><strong>Additional models and CCTV evaluation</strong><p>Compare models and assess performance on suitable surveillance-like data.</p><small>PLANNED</small></div></div>
        </div>
      </div>
    </>
  );
}

function About() {
  return (
    <>
      <SectionHeading eyebrow="PROJECT / INFORMATION" title="About the project" description="Research-oriented image super-resolution framework for low-quality surveillance imagery." />
      <div className="panel about-hero"><div className="about-icon"><ScanEye size={34} /></div><div><div className="eyebrow">B.TECH AI / ML PROJECT</div><h2>Development of a Deep Learning Image Super-Resolution Framework</h2><p>Engineer an image super-resolution framework for enhancing low-quality surveillance footage using deep learning and image processing.</p></div></div>
      <div className="content-grid">
        <div className="panel"><h3>Technology stack</h3><div className="tech-list">{["Python", "PyTorch", "SRCNN", "OpenCV", "React", "Vite", "FastAPI (planned)"].map(t => <span key={t}>{t}</span>)}</div></div>
        <div className="panel"><h3>Current model architecture</h3><div className="architecture-list"><div><span>Layer 1</span><strong>9×9 convolution · 3 → 64</strong></div><div><span>Layer 2</span><strong>1×1 convolution · 64 → 32</strong></div><div><span>Layer 3</span><strong>5×5 convolution · 32 → 3</strong></div><div><span>Parameters</span><strong>20,099</strong></div></div></div>
      </div>
      <div className="warning-box"><ShieldCheck size={19} /><div><strong>Responsible interpretation</strong><p>Super-resolution estimates missing high-frequency information and may hallucinate plausible detail. Outputs must not be represented as recovered ground truth or used alone to identify people, read number plates, or establish forensic facts. Keep originals available and label enhanced images clearly.</p></div></div>
    </>
  );
}

export default function App() {
  const [active, setActive] = useState("overview");
  const [mobileNav, setMobileNav] = useState(false);
  const [file, setFile] = useState(null);
  const [preview, setPreview] = useState("");

  useEffect(() => () => { if (preview) URL.revokeObjectURL(preview); }, [preview]);

  const page = {
    overview: <Overview navigate={setActive} />,
    enhance: <Enhancement file={file} setFile={setFile} preview={preview} setPreview={setPreview} />,
    performance: <Performance />,
    dataset: <Dataset />,
    experiments: <Experiments />,
    about: <About />,
  }[active];

  const activeLabel = navItems.find(n => n.id === active)?.label || "Overview";

  return (
    <div className="app-shell">
      {mobileNav && <button className="mobile-scrim" onClick={() => setMobileNav(false)} aria-label="Close navigation" />}
      <aside className={`sidebar ${mobileNav ? "sidebar-open" : ""}`}>
        <div className="brand">
          <div className="brand-mark"><ScanEye size={23} /></div>
          <div><strong>SR<span>STUDIO</span></strong><small>RESEARCH CONSOLE</small></div>
          <button className="mobile-close icon-btn" onClick={() => setMobileNav(false)}><X size={18} /></button>
        </div>
        <div className="project-switcher"><div className="project-avatar">SR</div><div><strong>Surveillance SR</strong><span>Research project</span></div><ChevronRight size={15} /></div>
        <nav className="sidebar-nav">
          {["WORKSPACE", "ANALYTICS", "PROJECT"].map(group => (
            <div className="nav-group" key={group}>
              <div className="nav-label">{group}</div>
              {navItems.filter(item => item.group === group).map(item => {
                const Icon = item.icon;
                return <button key={item.id} className={`nav-item ${active === item.id ? "active" : ""}`} onClick={() => { setActive(item.id); setMobileNav(false); }}><Icon size={18} /><span>{item.label}</span>{active === item.id && <span className="nav-active-dot" />}</button>;
              })}
            </div>
          ))}
        </nav>
        <div className="sidebar-bottom">
          <div className="system-card"><div className="system-card-top"><span className="live-dot" /> SYSTEM STATUS</div><strong>Frontend operational</strong><small>Model API not connected</small><div className="system-track"><span /></div></div>
          <div className="sidebar-footer"><span>SR STUDIO · v1.0.0</span><span>Built for research</span></div>
        </div>
      </aside>

      <main className="main-area">
        <header className="topbar">
          <div className="topbar-left"><button className="mobile-menu icon-btn" onClick={() => setMobileNav(true)}><Menu size={20} /></button><div className="breadcrumb">Workspace <ChevronRight size={14} /> <strong>{activeLabel}</strong></div></div>
          <div className="topbar-right"><span className="topbar-status"><span className="live-dot" /> Local environment</span><button className="icon-btn" title="Help"><CircleHelp size={18} /></button><div className="user-avatar">AI</div></div>
        </header>
        <div className="page-content" key={active}>{page}</div>
        <footer className="app-footer"><span>© 2026 Surveillance SR Studio</span><span>Research prototype <i /> SRCNN · DIV2K</span></footer>
      </main>
    </div>
  );
}


