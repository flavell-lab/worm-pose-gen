// The Recordings screen (docs/APP_SIMPLIFICATION.md, section 2): the chosen
// setup's recordings with their status, and Open on every row (it makes the
// recording's workspace when it has none); the analysis starts inside the
// workspace. While an analysis is queued or running its row also has Cancel.
// Two tables, each recording in one: those analysed or analysing (or whose
// analysis failed or was cancelled), most recently opened first, then those
// not analysed yet. The recordings are those of the user's recording folders
// (searched two folders deep for .h5, .avi and .mp4 files) and single files
// added to a setup; Add recordings adds or removes either. Before the user has any
// recording the screen says what to do first and where their data goes. New
// files are listed at once and opened by the server in the background ("Checking
// the file…") to read their length and whether they can be read. The
// tables rescan whenever the screen opens and poll while an analysis runs; a
// poll rebuilds only the rows that changed, so the scroll position stays.

import {api, el, post, query} from "./api.js";

const POLL_MS = 3000;
const SETUP_KEY = "workspace.setup";
// The states of the "Analysed and analysing" table: every state an analysis leads to.
const RECENT_STATES = new Set(["queued", "analysing", "failed", "cancelled", "analysed", "issues", "reviewed", "exported"]);

function remembered(key) { try { return localStorage.getItem(key) || ""; } catch { return ""; } }
function remember(key, value) { try { localStorage.setItem(key, value); } catch { /* storage blocked */ } }

export function formatDuration(frames, fps) {
  if (!frames || !fps) return "";
  const seconds = Math.round(frames / fps), m = Math.floor(seconds / 60), s = seconds % 60;
  return `${m}:${String(s).padStart(2, "0")}`;
}

export function formatWhen(iso) {
  if (!iso) return "—";
  const when = new Date(iso), days = (Date.now() - when.getTime()) / 864e5;
  if (days < 1 && when.getDate() === new Date().getDate()) return when.toLocaleTimeString([], {hour: "2-digit", minute: "2-digit"});
  return when.toLocaleDateString([], {month: "short", day: "numeric", year: days > 300 ? "numeric" : undefined});
}

// The status cell of a recording (and the Workspace header's progress): [text, kind, progress | null].
export function describeStatus(status) {
  if (!status) return ["Not analysed", "", null];
  const job = status.analysis;
  switch (status.state) {
    case "queued": return [job?.slurm_state === "PENDING" ? "Waiting for SLURM" : "Queued", "info", null];
    case "analysing": return [`Analysing · ${stageOf(job)}`, "info", job?.progress ?? 0];
    case "failed": return ["Analysis failed", "error", null];
    case "cancelled": return ["Analysis cancelled", "", null];
    case "not_analysed": return ["Not analysed", "", null];
    case "analysed": return ["Analysed · open to review", "ok", null];
    case "issues": return [`${status.issues.unreviewed} issue${status.issues.unreviewed === 1 ? "" : "s"} to review`, "warn", null];
    case "reviewed": return ["Reviewed", "ok", null];
    case "exported": return ["Exported", "ok", null];
    default: return [status.state, "", null];
  }
}

export const STAGE_WORDS = {segment: "masks", prior: "body size", fit: "fitting", ambiguity: "checking", propagate: "tracking", track: "lengths", fixed_body: "fixed body"};
export function stageOf(job) {
  const stage = String(job?.message || "").split(":")[0];
  return `${STAGE_WORDS[stage] || "starting"} ${Math.round((job?.progress || 0) * 100)}%`;
}

export class RecordingsScreen {
  constructor(section, ctx, {onOpen}) {
    this.ctx = ctx;
    this.onOpen = onOpen;
    this.data = null;
    this.filter = "";
    this.timer = null;
    this.node = el("div", {class: "ws-home", hidden: true});
    section.append(this.node);
  }

  async show() {
    this.node.hidden = false;
    this.ctx.setHeader("workspace", {context: [el("span", {class: "ws-title"}, "Recordings")]});
    await this.refresh();
  }

  hide() {
    this.node.hidden = true;
    clearTimeout(this.timer);
  }

  async refresh() {
    clearTimeout(this.timer);
    const setup = this.data?.setup?.ref || remembered(SETUP_KEY);
    try {
      this.data = await api(`/api/home${query({setup})}`);
    } catch (error) {
      if (setup && error.status === 404) { remember(SETUP_KEY, ""); this.data = await api("/api/home"); }
      else { this.renderError(error); return; }
    }
    if (this.tables && this.renderedSetup === this.data.setup?.ref) this.renderRows(); else this.render();
    // Poll while an analysis runs or while the server is still opening new files to read their length.
    const busy = this.data.recordings.some((r) => r.pending || ["queued", "analysing"].includes(r.status?.state));
    if (busy && !this.node.hidden) this.timer = setTimeout(() => this.refresh(), POLL_MS);
  }

  renderError(error) {
    this.node.replaceChildren(el("div", {class: "empty"}, el("p", {}, "Could not list the recordings."), el("p", {class: "note error"}, error.message),
      el("button", {onclick: () => this.refresh()}, "Try again")));
  }

  render() {
    const {setups, setup} = this.data;
    this.tables = null; this.renderedSetup = setup?.ref ?? null; this.rows = new Map(); this.thumbs = new Map();
    if (!setup) {
      const libraries = this.ctx.config?.libraries || {};
      this.node.replaceChildren(el("div", {class: "empty ws-empty"},
        el("h2", {}, "No microscope setups yet"),
        el("p", {}, "A setup says how to read a microscope's videos and which models to use on them. Setups come from the lab library or your own."),
        el("p", {class: "note"}, `Lab library: ${libraries.lab || "none on this machine"}${libraries.lab && !libraries.lab_available ? " (not reachable)" : ""} · yours: ${libraries.personal || "?"}`)));
      return;
    }
    const setupControl = setups.length > 1
      ? el("label", {class: "inline"}, "Setup", el("select", {onchange: (event) => { remember(SETUP_KEY, event.target.value); this.data.setup = {ref: event.target.value}; this.refresh(); }},
        setups.map((s) => el("option", {value: s.ref, selected: s.ref === setup.ref}, s.name))))
      : el("span", {class: "ws-setup-name", title: setup.ref}, setup.name);
    const facts = [setup.pixel_size_um ? `${+setup.pixel_size_um.toPrecision(3)} µm/px` : null, setup.fps ? `${setup.fps} fps` : null].filter(Boolean).join(" · ");
    const toolbar = el("div", {class: "ws-home-bar"},
      setupControl, facts ? el("span", {class: "note"}, facts) : null,
      el("span", {class: "ws-grow"}),
      el("input", {type: "search", class: "ws-filter", placeholder: "Filter recordings", value: this.filter, oninput: (event) => { this.filter = event.target.value; this.renderRows(); }}),
      el("button", {onclick: () => this.addRecordings()}, "Add recordings…"),
    );
    if (!this.data.recordings.length && !this.data.folders.length) {
      this.node.replaceChildren(toolbar, this.welcome());
      return;
    }
    const table = (tbody) => el("div", {class: "ws-table-wrap"}, el("table", {class: "data ws-recordings"},
      el("thead", {}, el("tr", {}, el("th", {class: "ws-thumb-cell"}), el("th", {}, "Recording"), el("th", {class: "num"}, "Length"),
        el("th", {}, "Status"), el("th", {}, "Last opened"), el("th", {class: "ws-action-cell"}))),
      tbody));
    const recent = el("tbody"), rest = el("tbody");
    this.recentSection = el("section", {class: "ws-section"}, el("h2", {class: "ws-section-title"}, "Analysed and analysing"), table(recent));
    this.restSection = el("section", {class: "ws-section"}, el("h2", {class: "ws-section-title"}, "Not analysed"), table(rest));
    this.tables = {recent, rest};
    this.node.replaceChildren(toolbar, el("div", {class: "ws-lists"}, this.recentSection, this.restSection));
    this.renderRows();
  }

  renderRows() {
    if (!this.tables) return;
    const needle = this.filter.trim().toLowerCase(), all = this.data.recordings;
    const matches = (r) => !needle || r.id.toLowerCase().includes(needle);
    const analysed = all.filter((r) => RECENT_STATES.has(r.status?.state)), rest = all.filter((r) => !RECENT_STATES.has(r.status?.state));
    analysed.sort((a, b) => (b.status.last_opened || "").localeCompare(a.status.last_opened || ""));
    // A table shows only when it has recordings (the second one also when the setup has none, to say so).
    this.recentSection.hidden = !analysed.length;
    this.restSection.hidden = !rest.length && !!all.length;
    this.fill("recent", analysed.filter(matches), "No analysed recording matches the filter.");
    this.fill("rest", rest.filter(matches), all.length ? "No recording matches the filter."
      : `No .h5, .avi or .mp4 files in ${this.data.folders.length === 1 ? "the folder" : "the folders"} of this setup (searched two folders deep). Add another with Add recordings.`);
  }

  // Put the rows in a table, reusing the row of a recording whose data has not changed since the last poll.
  fill(table, recordings, emptyText) {
    const fps = this.data.setup.fps, tbody = this.tables[table], seen = new Set();
    const nodes = recordings.map((recording) => {
      const key = `${table}:${recording.id}`, stamp = JSON.stringify(recording), cached = this.rows.get(key);
      seen.add(key);
      if (cached?.stamp === stamp) return cached.node;
      const node = this.row(recording, fps, key);
      this.rows.set(key, {stamp, node});
      return node;
    });
    for (const key of this.rows.keys()) if (key.startsWith(`${table}:`) && !seen.has(key)) this.rows.delete(key);
    if (!nodes.length) nodes.push(el("tr", {}, el("td", {colspan: 6, class: "empty"}, emptyText)));
    const current = [...tbody.children];
    if (current.length !== nodes.length || current.some((node, k) => node !== nodes[k])) tbody.replaceChildren(...nodes);
  }

  // A row's thumbnail is kept across rebuilds of the row, so a poll does not reload it.
  thumbnail(key, recording) {
    if (!recording.readable || recording.pending) return el("span", {class: "ws-thumb"});
    const src = `/api/recordings/thumbnail${query({path: recording.path, frame: Math.floor((recording.frames || 1) / 2), scale: 0.12})}`;
    const cached = this.thumbs.get(key);
    if (cached?.dataset.src === src) return cached;
    const image = el("img", {class: "ws-thumb", loading: "lazy", alt: "", src, dataset: {src}});
    this.thumbs.set(key, image);
    return image;
  }

  row(recording, fps, key) {
    const status = recording.status;
    const [text, kind, progress] = recording.pending ? ["Checking the file…", "info", null]
      : !recording.readable ? ["Cannot be read", "error", null] : describeStatus(status);
    const thumb = this.thumbnail(key, recording);
    const statusCell = el("td", {class: "ws-status"}, el("span", {class: `badge ${kind}`, title: status?.analysis?.error || recording.error || null}, text));
    if (progress !== null) statusCell.append(el("progress", {max: 1, value: progress}));
    const running = ["queued", "analysing"].includes(status?.state);
    const actions = el("div", {class: "ws-rec-actions"},
      running ? el("button", {title: "Stop the analysis", onclick: (event) => this.cancel(recording, event.currentTarget)}, "Cancel") : null,
      el("button", {class: "primary", disabled: !recording.readable, onclick: (event) => this.open(recording, event.currentTarget)}, "Open"));
    return el("tr", {dataset: {recording: recording.id}},
      el("td", {class: "ws-thumb-cell"}, thumb),
      el("td", {}, el("div", {class: "ws-rec-name"}, recording.id), el("div", {class: "note ws-rec-path", title: recording.path}, recording.path)),
      el("td", {class: "num"}, recording.frames ? `${recording.frames.toLocaleString()} fr` : "—", fps && recording.frames ? el("div", {class: "note"}, formatDuration(recording.frames, fps)) : null),
      statusCell,
      el("td", {class: "note"}, formatWhen(status?.last_opened)),
      el("td", {class: "ws-action-cell"}, actions),
    );
  }

  // Open the recording's workspace, making it first when the recording has none.
  async open(recording, button) {
    if (recording.workspace) { this.onOpen(recording.workspace); return; }
    button.disabled = true;
    try {
      this.onOpen((await post("/api/recordings/open", {path: recording.path})).workspace);
    } catch (error) {
      this.ctx.toast(error.message, "error");
      button.disabled = false;
    }
  }

  async cancel(recording, button) {
    button.disabled = true;
    try {
      await post(`/api/jobs/${recording.status.analysis.id}/cancel`);
      this.ctx.toast(`Cancelled the analysis of ${recording.id}`, "ok");
    } catch (error) {
      this.ctx.toast(error.message, "error");
    }
    await this.refresh();
  }

  // The first screen of a new user: what to do first, and where their work is saved.
  welcome() {
    const config = this.ctx.config || {}, libraries = config.libraries || {};
    return el("div", {class: "empty ws-empty"},
      el("h2", {}, "Add your recordings"),
      el("p", {}, "Choose the folder your recordings are in. Every .h5 recording and .avi or .mp4 video in it, and in the folders up to two levels below it, is listed here."),
      el("p", {}, el("button", {class: "primary", onclick: () => this.addRecordings()}, "Add a folder…")),
      el("p", {class: "note"}, `Your labels, models and analyses are saved in ${config.data_dir || "?"}. To use another folder, run scripts/bootstrap_environment.sh --data-dir <folder> and start the app again.`),
      el("p", {class: "note"}, libraries.lab_available ? `Lab models and labels: ${libraries.lab} (read-only).`
        : `The lab library ${libraries.lab ? `at ${libraries.lab} cannot be read from this machine` : "is not known on this machine"}: only your own models and labels are available.`));
  }

  addRecordings() {
    openAddRecordings(this.ctx, this.data.setups, this.data.setup.ref).then((changed) => {
      if (!changed) return;
      if (changed.setup !== this.data.setup.ref) { remember(SETUP_KEY, changed.setup); this.data.setup = {ref: changed.setup}; }
      this.tables = null;
      this.refresh();
    });
  }
}


// Add recordings: the recording folders with Remove, a path box with a browser of folders, HDF5 files and videos,
// and the setup a new folder or file belongs to. "Add this folder" adds the folder shown (searched two folders
// deep); "Add file" registers one file (an HDF5 recording or a video).
// Resolves to {setup} when anything changed.
function openAddRecordings(ctx, setups, current) {
  return new Promise((resolve) => {
    let changed = null, listing = null;
    const dialog = el("dialog", {class: "ws-dialog ws-add"});
    const folderTitle = el("h3", {class: "ws-section-title", hidden: true}, "Recording folders");
    const folders = el("div", {class: "list ws-folders"});
    const path = el("input", {type: "text", placeholder: "/path/to/folder, recording.h5 or video.avi", "aria-label": "Folder or recording path"});
    const list = el("div", {class: "list ws-browse"});
    const crumbs = el("div", {class: "note ws-browse-at"});
    const error = el("p", {class: "note error", hidden: true});
    const setup = el("select", {"aria-label": "Setup"}, setups.map((s) => el("option", {value: s.ref, selected: s.ref === current}, s.name)));
    const addFolder = el("button", {type: "button", disabled: true}, "Add this folder");
    const addFile = el("button", {class: "primary", type: "button", disabled: true}, "Add file");
    const isFile = (value) => /\.(h5|hdf5|avi|mp4)$/i.test(value);
    const fail = (failure) => { error.textContent = failure.message; error.hidden = false; };
    const setupName = (ref) => setups.find((s) => s.ref === ref)?.name || ref;
    const showFolders = async () => {
      const {folders: rows} = await api("/api/library/recording-folders");
      folderTitle.hidden = !rows.length;
      folders.replaceChildren(...rows.map((folder) => el("div", {class: "item ws-folder"},
        el("span", {class: "ws-folder-path", title: folder.path}, folder.path), el("span", {class: "note"}, setupName(folder.setup)),
        el("button", {type: "button", title: "Stop listing this folder's recordings; their workspaces and labels stay", onclick: async () => {
          try {
            await post("/api/library/recording-folders/remove", {path: folder.path});
            changed = {setup: setup.value};
            await showFolders();
          } catch (failure) { fail(failure); }
        }}, "Remove"))));
    };
    const browse = async (at) => {
      try {
        listing = await api(`/api/files${query({path: at})}`);
      } catch (failure) { fail(failure); return; }
      error.hidden = true;
      crumbs.textContent = listing.path;
      path.value = listing.path;
      addFolder.disabled = false; addFile.disabled = true;
      const entries = listing.entries.filter((entry) => ["dir", "h5", "video"].includes(entry.kind));
      list.replaceChildren(
        ...(listing.parent ? [el("div", {class: "item", onclick: () => browse(listing.parent)}, "↑ ..")] : []),
        ...entries.map((entry) => el("div", {class: "item", onclick: () => {
          if (entry.kind === "dir") browse(entry.path);
          else { path.value = entry.path; addFile.disabled = false; addFolder.disabled = true; }
        }}, entry.kind === "dir" ? `▸ ${entry.name}` : entry.name, entry.kind !== "dir" && entry.size_bytes ? el("span", {class: "note"}, ` ${(entry.size_bytes / 1e9).toFixed(1)} GB`) : null)),
      );
    };
    path.addEventListener("input", () => { const value = path.value.trim(); addFile.disabled = !isFile(value); addFolder.disabled = !value || isFile(value); });
    path.addEventListener("keydown", (event) => {
      if (event.key !== "Enter") return;
      const value = path.value.trim();
      if (isFile(value)) addFile.click(); else if (value) browse(value);
    });
    addFolder.addEventListener("click", async () => {
      addFolder.disabled = true;
      try {
        const added = await post("/api/library/recording-folders", {path: path.value.trim(), setup: setup.value});
        changed = {setup: added.setup};
        ctx.toast(`Listing the recordings in ${added.path}`, "ok");
        dialog.close();
      } catch (failure) { fail(failure); addFolder.disabled = false; }
    });
    addFile.addEventListener("click", async () => {
      addFile.disabled = true;
      try {
        const added = await post("/api/library/recordings", {path: path.value.trim(), setup: setup.value});
        changed = {setup: added.setup};
        ctx.toast(`Added ${added.id}`, "ok");
        dialog.close();
      } catch (failure) { fail(failure); addFile.disabled = false; }
    });
    dialog.append(
      el("h2", {}, "Add recordings"),
      folderTitle, folders,
      el("p", {class: "note"}, "Add a folder to list every .h5 recording and .avi or .mp4 video in it and up to two folders below it, or add a single file."),
      el("div", {class: "ws-form-row"}, path, el("button", {type: "button", onclick: () => browse(path.value.trim() || undefined)}, "Browse")),
      crumbs, list, error,
      el("div", {class: "ws-form-row"}, el("span", {class: "ws-form-label"}, "Setup"), setup),
      el("div", {class: "ws-dialog-actions"}, el("button", {type: "button", onclick: () => dialog.close()}, "Close"), addFolder, addFile),
    );
    dialog.addEventListener("close", () => { dialog.remove(); resolve(changed); });
    document.body.append(dialog);
    dialog.showModal();
    showFolders().catch(fail);
    path.focus();
  });
}
