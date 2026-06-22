// ─────────────────────────────────────────────────────────────────────────────
//  MODE SWITCHING  (image / video / live — mutually exclusive)
// ─────────────────────────────────────────────────────────────────────────────

let activeMode = "image";
let livePollingActive = false;
let videoProgressInterval = null;
let latest_data_output_file = "";

function switchMode(mode) {
    if (mode === activeMode) return;

    // stop live feed if switching away
    if (activeMode === "live") stopLiveFeed();

    document.querySelectorAll(".mode-tab").forEach(t => {
        t.classList.toggle("active", t.dataset.mode === mode);
    });
    document.querySelectorAll(".mode-panel").forEach(p => {
        p.classList.add("d-none");
    });
    document.getElementById("panel-" + mode).classList.remove("d-none");

    activeMode = mode;
}

document.querySelectorAll(".mode-tab").forEach(tab => {
    tab.addEventListener("click", () => switchMode(tab.dataset.mode));
});


// ─────────────────────────────────────────────────────────────────────────────
//  DASHBOARD POLLING
// ─────────────────────────────────────────────────────────────────────────────

async function updateDashboard() {
    try {
        const res  = await fetch("/detections");
        const data = await res.json();

        setText("fps",        data.fps);
        setText("confidence", data.confidence + "%");
        setText("totalSigns", data.total_signs);

        setText("analyticsTotalSigns",  data.total_signs);
        setText("analyticsFps",         data.fps + " FPS");
        setText("analyticsConfidence",  data.avg_confidence + "%");
        setText("topSign",              data.top_sign || "NONE");
        setText("historyCount",         data.history.length);
        setText("alertCount",           data.alerts);

        // live current-signs panel (only shown in live mode)
        const signsEl = document.getElementById("detectedSigns");
        if (signsEl) {
            signsEl.innerHTML = data.current_signs.length === 0
                ? `<li class="incident-item" style="color:var(--text-lo);text-align:center;">No signs in frame</li>`
                : data.current_signs.map(s => `<li class="incident-item">${s}</li>`).join("");
        }

        // history list (upgraded)
        renderHistory(data.history);

    } catch (err) {
        console.warn("Dashboard poll error:", err);
    }
}


// ─────────────────────────────────────────────────────────────────────────────
//  VIOLATIONS POLLING
// ─────────────────────────────────────────────────────────────────────────────

async function updateViolations() {
    try {
        const res        = await fetch("/violations");
        const violations = await res.json();
        const listEl     = document.getElementById("violationList");
        const countEl    = document.getElementById("violationCount");
        if (!listEl) return;

        if (countEl) countEl.textContent = violations.length + " incident" + (violations.length !== 1 ? "s" : "");

        if (violations.length === 0) {
            listEl.innerHTML = `
                <li class="violation-empty">
                    <span class="violation-empty-icon">🛡️</span>
                    <span>No violations detected yet</span>
                </li>`;
            return;
        }

        const SEV_CLASS = { HIGH: "sev-high", MEDIUM: "sev-mid", LOW: "sev-low" };

        listEl.innerHTML = violations.map(v => `
            <li class="violation-item ${SEV_CLASS[v.severity] || ""}">
                <span class="violation-sev-badge">${v.severity}</span>
                <span class="violation-sign-name">${v.sign}</span>
                <span class="violation-time-badge mono">${v.time}</span>
            </li>`).join("");

    } catch (err) {
        console.warn("Violations poll error:", err);
    }
}


// ─────────────────────────────────────────────────────────────────────────────
//  HISTORY RENDER
// ─────────────────────────────────────────────────────────────────────────────

function renderHistory(history) {
    const el = document.getElementById("historyList");
    const totalEl = document.getElementById("historyTotal");
    if (!el) return;

    if (totalEl) totalEl.textContent = history.length + " entr" + (history.length === 1 ? "y" : "ies");

    if (history.length === 0) {
        el.innerHTML = `<li class="history-item" style="justify-content:center; color:var(--text-lo);">No detections yet</li>`;
        return;
    }

    el.innerHTML = history.map((item, i) => `
        <li class="history-item">
            <span class="history-index">${String(i + 1).padStart(2, "0")}</span>
            <span class="history-dot"></span>
            <span class="history-sign-name">${item.sign}</span>
            <span class="history-ts">${item.time}</span>
        </li>`).join("");
}


// ─────────────────────────────────────────────────────────────────────────────
//  IMAGE DETECTION
// ─────────────────────────────────────────────────────────────────────────────

document.addEventListener("DOMContentLoaded", () => {

    const imageInput = document.getElementById("imageFileInput");
    if (imageInput) {
        imageInput.addEventListener("change", async () => {
            const file = imageInput.files[0];
            if (!file) return;

            const panel = document.getElementById("imageResultPanel");
            const img   = document.getElementById("imageResultImg");
            const signs = document.getElementById("imageResultSigns");

            panel.classList.remove("d-none");
            img.src = "";
            signs.innerHTML = `<li class="incident-item" style="color:var(--text-lo)">Processing…</li>`;

            const fd = new FormData();
            fd.append("file", file);

            try {
                const res  = await fetch("/detect/image", { method: "POST", body: fd });
                const data = await res.json();

                if (data.error) {
                    signs.innerHTML = `<li class="incident-item" style="color:var(--sign-red)">${data.error}</li>`;
                    return;
                }

                img.src = "data:image/jpeg;base64," + data.image;

                signs.innerHTML = data.signs.length === 0
                    ? `<li class="incident-item" style="color:var(--text-lo);text-align:center;">No signs detected</li>`
                    : data.signs.map(s => `<li class="incident-item">${s}</li>`).join("");

            } catch (err) {
                signs.innerHTML = `<li class="incident-item" style="color:var(--sign-red)">Upload failed</li>`;
                console.error("Image detection error:", err);
            }

            imageInput.value = "";
        });
    }

    // clear button for image
    const imageClearBtn = document.getElementById("imageClearBtn");
    if (imageClearBtn) {
        imageClearBtn.addEventListener("click", () => {
            document.getElementById("imageResultPanel").classList.add("d-none");
            document.getElementById("imageResultImg").src = "";
            document.getElementById("imageResultSigns").innerHTML = "";
        });
    }

    // ── VIDEO DETECTION ─────────────────────────────────────────────────────────────
    // POST /detect/video returns immediately with the output filename.
    // ALL "done" logic is driven by polling /detect/video/progress only.

    const videoInput = document.getElementById("videoFileInput");
    if (videoInput) {
        videoInput.addEventListener("change", async () => {
            const file = videoInput.files[0];
            if (!file) return;
            videoInput.value = "";

            const panel  = document.getElementById("videoResultPanel");
            const stream = document.getElementById("videoResultStream");
            const hudTag = document.getElementById("videoHudTag");
            const dlWrap = document.getElementById("videoDownloadWrap");
            const dlBtn  = document.getElementById("videoDownloadBtn");

            // Reset UI
            panel.classList.remove("d-none");
            dlWrap.classList.add("d-none");
            if (hudTag) hudTag.textContent = "PROCESSING";
            setVideoProgress(0, 0);
            if (stream) stream.style.display = "none";

            const oldPlayer = document.getElementById("videoResultPlayer");
            if (oldPlayer) { oldPlayer.pause(); oldPlayer.removeAttribute("src"); oldPlayer.load(); oldPlayer.style.display = "none"; }

            // Show spinner
            const hudFrame = stream?.closest(".hud-frame") || document.querySelector("#panel-video .hud-frame");
            let spinner = document.getElementById("videoSpinner");
            if (!spinner && hudFrame) {
                spinner = document.createElement("div");
                spinner.id = "videoSpinner";
                spinner.style.cssText = "display:flex;align-items:center;justify-content:center;min-height:260px;flex-direction:column;gap:14px;";
                spinner.innerHTML = `<div class="video-spinner"></div><span style="font-family:'IBM Plex Mono',monospace;font-size:0.8rem;letter-spacing:0.08em;color:var(--amber);">Processing frames…</span>`;
                hudFrame.appendChild(spinner);
            } else if (spinner) {
                spinner.style.display = "flex";
            }

            if (videoProgressInterval) { clearInterval(videoProgressInterval); videoProgressInterval = null; }

            // Upload — server queues job and returns immediately
            let expectedFilename = "";
            try {
                const fd = new FormData();
                fd.append("file", file);
                const res  = await fetch("/detect/video", { method: "POST", body: fd });
                const data = await res.json();
                if (!res.ok || data.error) {
                    console.error("Video upload failed:", data.error);
                    if (hudTag) hudTag.textContent = "ERROR";
                    const sp = document.getElementById("videoSpinner");
                    if (sp) sp.style.display = "none";
                    return;
                }
                expectedFilename = data.output_file || "";
            } catch (err) {
                console.error("Video upload error:", err);
                if (hudTag) hudTag.textContent = "ERROR";
                const sp = document.getElementById("videoSpinner");
                if (sp) sp.style.display = "none";
                return;
            }

            // Poll until background job is truly done.
            // Require pollCount >= 2 and frames_done > 0 before trusting done=true,
            // to avoid stale state from a previous run.
            let pollCount = 0;
            videoProgressInterval = setInterval(async () => {
                try {
                    const res  = await fetch("/detect/video/progress");
                    const prog = await res.json();

                    setVideoProgress(prog.frames_done, prog.frames_total);
                    if (prog.output_file) latest_data_output_file = prog.output_file;
                    pollCount++;

                    const reallyDone = prog.done && prog.frames_done > 0 && pollCount >= 2;
                    if (!reallyDone) return;

                    clearInterval(videoProgressInterval);
                    videoProgressInterval = null;

                    const fname = expectedFilename || prog.output_file || latest_data_output_file;
                    if (hudTag) hudTag.textContent = "DONE";

                    const sp = document.getElementById("videoSpinner");
                    if (sp) sp.style.display = "none";

                    // Build video player
                    const hf = document.querySelector("#panel-video .hud-frame");
                    let videoEl = document.getElementById("videoResultPlayer");
                    if (!videoEl && hf) {
                        videoEl = document.createElement("video");
                        videoEl.id = "videoResultPlayer";
                        videoEl.controls = true;
                        videoEl.style.cssText = "width:100%;display:block;border-radius:0;min-height:260px;background:#0D0F12;";
                        hf.appendChild(videoEl);
                    }
                    if (videoEl && fname) {
                        videoEl.style.display = "block";
                        videoEl.src = "/preview/video/" + fname + "?t=" + Date.now();
                        videoEl.load();
                    }

                    // Wire download button
                    if (fname && dlBtn) {
                        dlBtn.onclick = async (e) => {
                            e.preventDefault();
                            try {
                                const r = await fetch("/download/video/" + fname);
                                if (!r.ok) { alert("File not ready yet, please try again."); return; }
                                const blob = await r.blob();
                                const a = document.createElement("a");
                                a.href = URL.createObjectURL(blob);
                                a.download = fname;
                                document.body.appendChild(a);
                                a.click();
                                document.body.removeChild(a);
                                setTimeout(() => URL.revokeObjectURL(a.href), 5000);
                            } catch (err) {
                                alert("Download failed: " + err.message);
                            }
                        };
                        dlWrap.classList.remove("d-none");
                    }

                } catch (pollErr) {
                    console.warn("Progress poll error (will retry):", pollErr);
                }
            }, 1000);
        });
    }

    // clear button for video
    const videoClearBtn = document.getElementById("videoClearBtn");
    if (videoClearBtn) {
        videoClearBtn.addEventListener("click", () => {
            document.getElementById("videoResultPanel").classList.add("d-none");
            if (videoProgressInterval) { clearInterval(videoProgressInterval); videoProgressInterval = null; }
            setVideoProgress(0, 0);
            const vp = document.getElementById("videoResultPlayer");
            if (vp) { vp.src = ""; vp.style.display = "none"; }
            const sp = document.getElementById("videoSpinner");
            if (sp) sp.style.display = "none";
            const st = document.getElementById("videoResultStream");
            if (st) st.style.display = "none";
        });
    }

    // ── LIVE FEED ────────────────────────────────────────────────────────────

    const startLiveBtn = document.getElementById("startLiveBtn");
    if (startLiveBtn) {
        startLiveBtn.addEventListener("click", startLiveFeed);
    }

    const stopLiveBtn = document.getElementById("stopLiveBtn");
    if (stopLiveBtn) {
        stopLiveBtn.addEventListener("click", stopLiveFeed);
    }

});


// ─────────────────────────────────────────────────────────────────────────────
//  VIDEO PROGRESS POLLING
// ─────────────────────────────────────────────────────────────────────────────

async function pollVideoProgress() {
    try {
        const res  = await fetch("/detect/video/progress");
        const data = await res.json();

        setVideoProgress(data.frames_done, data.frames_total);

        if (data.output_file) {
            latest_data_output_file = data.output_file;
        }

        if (data.done && videoProgressInterval) {
            clearInterval(videoProgressInterval);
            videoProgressInterval = null;
        }
    } catch (_) {}
}

function setVideoProgress(done, total) {
    const pct    = total > 0 ? Math.round((done / total) * 100) : 0;
    const bar    = document.getElementById("videoProgressBar");
    const pctEl  = document.getElementById("videoProgressPct");
    const label  = document.getElementById("videoProgressLabel");

    if (bar)   bar.style.width = pct + "%";
    if (pctEl) pctEl.textContent = pct + "%";
    if (label) label.textContent = done + " / " + total + " frames";
}


// ─────────────────────────────────────────────────────────────────────────────
//  LIVE FEED
// ─────────────────────────────────────────────────────────────────────────────

function startLiveFeed() {
    document.getElementById("liveStartWrap").classList.add("d-none");
    document.getElementById("livePanel").classList.remove("d-none");

    const img = document.getElementById("liveFeedImg");
    img.src = "/video";          // browser starts the MJPEG stream

    livePollingActive = true;
}

function stopLiveFeed() {
    livePollingActive = false;

    const img = document.getElementById("liveFeedImg");
    if (img) img.src = "";       // cut the stream

    document.getElementById("livePanel")?.classList.add("d-none");
    document.getElementById("liveStartWrap")?.classList.remove("d-none");
}


// ─────────────────────────────────────────────────────────────────────────────
//  HELPERS
// ─────────────────────────────────────────────────────────────────────────────

function setText(id, value) {
    const el = document.getElementById(id);
    if (el) el.innerText = value;
}

function indexOfBytes(haystack, needle) {
    outer: for (let i = 0; i <= haystack.length - needle.length; i++) {
        for (let j = 0; j < needle.length; j++) {
            if (haystack[i + j] !== needle[j]) continue outer;
        }
        return i;
    }
    return -1;
}


// ─────────────────────────────────────────────────────────────────────────────
//  POLLING INTERVALS
// ─────────────────────────────────────────────────────────────────────────────

setInterval(updateDashboard,  1000);
setInterval(updateViolations, 2000);

updateDashboard();
updateViolations();