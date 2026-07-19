/* ══════════════════════════════════════════════════════════════
   vRain Studio — app.js
══════════════════════════════════════════════════════════════ */

const API = "http://localhost:8766";

// ── 静态演示模式（GitHub Pages）────────────────────────────────
// 服务器未连接时，自动切换为静态模式：
// - 版式列表改用内置定义
// - 点「运行排版预览」时直接显示预置截图
// - 点「保存/新建/删除」时提示本地运行
const STATIC_PREVIEWS = {
  "24_black": ["preview_output/24_black_01.jpg", "preview_output/24_black_02.jpg"],
  "meipi_2":  ["preview_output/meipi_2_01.jpg",  "preview_output/meipi_2_02.jpg"],
  "meipi_3":  ["preview_output/meipi_3_01.jpg",  "preview_output/meipi_3_02.jpg"],
  "meipi_4":  ["preview_output/meipi_4_01.jpg",  "preview_output/meipi_4_02.jpg"],
  "meipi_5":  ["preview_output/meipi_5_01.jpg",  "preview_output/meipi_5_02.jpg"],
  "tu":       ["preview_output/tu_01.jpg",        "preview_output/tu_02.jpg"],
  "tu_2":     ["preview_output/tu_2_01.jpg",      "preview_output/tu_2_02.jpg"],
  "tu_3":     ["preview_output/tu_3_01.jpg",      "preview_output/tu_3_02.jpg"],
};

const STATIC_LAYOUT_DEFS = [
  { id:"24_black", desc:"24列·黑底经典" },
  { id:"meipi_2",  desc:"眉批版式 2" },
  { id:"meipi_3",  desc:"眉批版式 3" },
  { id:"meipi_4",  desc:"眉批版式 4" },
  { id:"meipi_5",  desc:"眉批版式 5" },
  { id:"tu",       desc:"图文排版" },
  { id:"tu_2",     desc:"图文排版 2" },
  { id:"tu_3",     desc:"图文排版 3" },
];

let STATIC_MODE = false;

function showLocalRunBanner() {
  const existing = document.getElementById("local-run-banner");
  if (existing) { existing.classList.add("flash"); setTimeout(()=>existing.classList.remove("flash"),600); return; }
  const banner = document.createElement("div");
  banner.id = "local-run-banner";
  banner.innerHTML = `
    <div class="lrb-icon">🖥</div>
    <div class="lrb-body">
      <div class="lrb-title">本功能需要在本地运行排版引擎</div>
      <div class="lrb-desc">在线演示仅展示预置效果图。完整功能请本地运行：</div>
      <pre class="lrb-code">git clone https://github.com/YOUR_REPO
cd train-pipeline-demo
pip install -r requirements.txt
python3 server.py</pre>
    </div>
    <button class="lrb-close" onclick="this.parentElement.remove()">✕</button>`;
  document.body.appendChild(banner);
  requestAnimationFrame(() => banner.classList.add("show"));
}

// ── 全局状态 ─────────────────────────────────────────────────
const S = {
  layouts: [],
  currentId: null,
  zoom: 0.8,
  annoZoom: 1.0,
  annoTool: "body",
  annoRects: [],   // [{tool, x, y, w, h}, ...]  body/margin/center/col_box
  annoLines: [],   // [{tool, x, y}, ...]         col/row（旧逻辑保留兼容）
  annoHSplit: null, // 横切线 y 坐标（原图坐标），切割正文/眉批
  annoDragging: false,
  annoStart: null,
  annoImg: null,
  annoDispScale: 1,
  annoCanvasId: "",   // 上传图片后自动设置，写入 canvas_id
  selectedImages: new Set(),  // 图文排版选中的图片素材
};

// ══════════════════════════════════════════════════════════════
//  工具函数
// ══════════════════════════════════════════════════════════════
function toast(msg, dur = 2500) {
  const t = document.getElementById("toast");
  t.textContent = msg; t.style.opacity = "1";
  clearTimeout(t._t);
  t._t = setTimeout(() => t.style.opacity = "0", dur);
}

async function api(method, path, data) {
  const opts = { method, headers: { "Content-Type": "application/json" } };
  if (data) opts.body = JSON.stringify(data);
  const r = await fetch(API + path, opts);
  if (!r.ok) {
    const e = await r.json().catch(() => ({ error: r.statusText }));
    throw new Error(e.error || r.statusText);
  }
  return r.json();
}

// ══════════════════════════════════════════════════════════════
//  服务器状态
// ══════════════════════════════════════════════════════════════
async function checkServer() {
  try {
    await api("GET", "/api/layouts");
    document.getElementById("server-dot").className = "dot-on";
    document.getElementById("server-label").textContent = "服务器已连接";
    STATIC_MODE = false;
    return true;
  } catch {
    document.getElementById("server-dot").className = "dot-off";
    document.getElementById("server-label").textContent = "演示模式（预置效果图）";
    STATIC_MODE = true;
    return false;
  }
}

// ══════════════════════════════════════════════════════════════
//  版式列表 & 加载
// ══════════════════════════════════════════════════════════════
async function loadLayouts() {
  if (STATIC_MODE) {
    // 静态模式：用内置定义构建版式列表（只有 id/name，cfg 用默认）
    S.layouts = STATIC_LAYOUT_DEFS.map(def => {
      const base = makeDefaultLayout();
      base.id = def.id; base.name = def.id;
      base.cfg.canvas_id = def.id;
      return base;
    });
    renderLayoutList();
    if (S.layouts.length > 0) selectLayout(S.layouts[0].id);
    return;
  }
  try {
    S.layouts = await api("GET", "/api/layouts");
  } catch {
    S.layouts = [makeDefaultLayout()];
    toast("无法连接服务器，使用内置默认版式");
  }
  renderLayoutList();
  if (S.layouts.length > 0) selectLayout(S.layouts[0].id);
}

function makeDefaultLayout() {
  const cfg = {
    canvas_id:"24_black", canvas_width:2480, canvas_height:1860,
    margins_top:200, margins_bottom:50, margins_left:50, margins_right:50,
    col_num:24, row_num:30, leaf_center_width:120,
    outline_width:10, outline_vmargin:5,
    spine_title:"", title_postfix:"",
    if_tpcenter:1, title_font_size:80, title_font_color:"black",
    title_y:1200, title_ydis:1.2, pager_font_size:35, pager_font_color:"black", pager_y:500,
    font1:"qiji-combo.ttf", font2:"HanaMinA.ttf", font3:"HanaMinB.ttf",
    text_fonts_array:"123", comment_fonts_array:"23",
    text_font1_size:60, text_font2_size:42, text_font3_size:42, text_font_color:"black",
    comment_font1_size:30, comment_font_color:"black",
    if_onlyperiod:1, onlyperiod_color:"red",
    if_book_vline:1, book_line_width:1, book_line_color:"black",
  };
  return { id: "24_black", name: "24_black", yaml: cfgToYAML(cfg, "24_black"), cfg };
}

function renderLayoutList() {
  const ul = document.getElementById("layout-list");
  ul.innerHTML = "";
  S.layouts.forEach(l => {
    const d = document.createElement("div");
    d.className = "li" + (l.id === S.currentId ? " active" : "");
    const c = l.cfg;
    d.innerHTML = `<div class="li-id">${l.id}</div>
      <div class="li-meta">${c.col_num||"?"}列×${c.row_num||"?"}行 · ${c.canvas_width||0}×${c.canvas_height||0}</div>`;
    d.onclick = () => selectLayout(l.id);
    ul.appendChild(d);
  });
  // 同步新建弹窗的「基于」下拉
  const sel = document.getElementById("new-base");
  if (sel) {
    sel.innerHTML = "";
    S.layouts.forEach(l => {
      const o = document.createElement("option");
      o.value = l.id; o.textContent = l.id; sel.appendChild(o);
    });
  }
}

function selectLayout(id) {
  S.currentId = id;
  clearDirty();   // 切换版式时清掉上一个的未保存标记
  document.querySelectorAll(".li").forEach(el => {
    el.classList.toggle("active", el.querySelector(".li-id").textContent === id);
  });
  const layout = S.layouts.find(l => l.id === id);
  if (!layout) return;
  document.getElementById("editing-id").textContent = id;
  renderParamForm(layout.cfg);
  updateStatusBar(layout.cfg);
  // 清空上次预览图
  document.getElementById("preview-pages").style.display = "none";
  document.getElementById("preview-empty").style.display = "";
}

// ══════════════════════════════════════════════════════════════
//  参数配置表单
// ══════════════════════════════════════════════════════════════
const SCHEMA = [
  { label:"📐 画布 & 留白", open:true, fields:[
    { k:"canvas_width",   label:"宽度 px",  type:"int" },
    { k:"canvas_height",  label:"高度 px",  type:"int" },
    { k:"margins_top",    label:"上留白",   type:"int" },
    { k:"margins_bottom", label:"下留白",   type:"int" },
    { k:"margins_left",   label:"左留白",   type:"int" },
    { k:"margins_right",  label:"右留白",   type:"int" },
  ]},
  { label:"📖 列行 & 版心", open:true, fields:[
    { k:"col_num",           label:"列数",       type:"int" },
    { k:"row_num",           label:"每列字数",   type:"int" },
    { k:"leaf_center_width", label:"版心宽 px",  type:"int" },
    { k:"outline_width",     label:"外框线宽",   type:"int" },
    { k:"outline_vmargin",   label:"外框纵向间距", type:"int" },
  ]},
  { label:"🏷 版心标题 & 页码", open:false, fields:[
    { k:"spine_title",    label:"版心标题",  type:"str" },
    { k:"title_postfix",  label:"卷号后缀",  type:"str" },
    { k:"if_tpcenter",    label:"标题居中",  type:"bool" },
    { k:"title_font_size",  label:"标题字号", type:"int" },
    { k:"title_font_color", label:"标题颜色", type:"color" },
    { k:"title_y",          label:"标题 Y",  type:"int" },
    { k:"pager_font_size",  label:"页码字号", type:"int" },
    { k:"pager_y",          label:"页码 Y",  type:"int" },
  ]},
  { label:"🖋 字体 & 字号", open:false, fields:[
    { k:"font1",  label:"字体1（正文主）", type:"font" },
    { k:"font2",  label:"字体2（备用）",   type:"font" },
    { k:"font3",  label:"字体3（备用）",   type:"font" },
    { k:"text_fonts_array",    label:"正文字体组", type:"str" },
    { k:"comment_fonts_array", label:"批注字体组", type:"str" },
    { k:"text_font1_size",     label:"正文字号1", type:"int" },
    { k:"text_font2_size",     label:"正文字号2", type:"int" },
    { k:"text_font_color",     label:"正文颜色",  type:"color" },
    { k:"comment_font1_size",  label:"批注字号（双排）",  type:"int" },
    { k:"comment_font_color",  label:"批注颜色（双排）",  type:"color" },
    { k:"topcmt_font_size",    label:"眉批字号",  type:"int" },
    { k:"topcmt_color",        label:"眉批颜色",  type:"color" },
  ]},
  { label:"📌 标点 & 书名竖线", open:false, fields:[
    { k:"if_onlyperiod",    label:"标点简化",  type:"bool" },
    { k:"onlyperiod_color", label:"句号颜色",  type:"color" },
    { k:"if_book_vline",    label:"书名竖线",  type:"bool" },
    { k:"book_line_width",  label:"竖线宽",    type:"int" },
    { k:"book_line_color",  label:"竖线颜色",  type:"color" },
  ]},
  { label:"🖼 图文排版", open:false, fields:[
    { k:"no_topcmt", label:"禁用眉批（无眉批区域时设1）", type:"bool" },
  ]},
];

// 字体列表（从服务器加载后填入）
let S_FONTS = [];

function renderParamForm(cfg) {
  const cont = document.getElementById("param-groups");
  cont.innerHTML = "";
  SCHEMA.forEach(group => {
    const wrap = document.createElement("div");
    wrap.className = "pg";
    const head = document.createElement("div");
    head.className = "pg-head";
    head.innerHTML = `<span>${group.label}</span><span>${group.open ? "▾":"▸"}</span>`;
    const body = document.createElement("div");
    body.className = "pg-body" + (group.open ? "" : " closed");

    group.fields.forEach(f => {
      const row = document.createElement("div");
      row.className = "pf";
      const lbl = document.createElement("label");
      lbl.textContent = f.label;
      row.appendChild(lbl);

      if (f.type === "bool") {
        const sel = document.createElement("select");
        [["1","是"],["0","否"]].forEach(([v,l]) => {
          const o = document.createElement("option");
          o.value = v; o.textContent = l;
          if (String(cfg[f.k] ?? "1") === v) o.selected = true;
          sel.appendChild(o);
        });
        sel.onchange = () => { cfg[f.k] = +sel.value; markDirty(); };
        row.appendChild(sel);
      } else if (f.type === "font") {
        const wrap2 = document.createElement("div");
        wrap2.style.cssText = "display:flex;gap:4px;flex:1;min-width:0";
        const sel = document.createElement("select");
        sel.style.flex = "1";
        const emptyOpt = document.createElement("option");
        emptyOpt.value = ""; emptyOpt.textContent = "（不使用）";
        sel.appendChild(emptyOpt);
        const curVal = cfg[f.k] || "";
        let matched = false;
        S_FONTS.forEach(fn => {
          const o = document.createElement("option");
          o.value = fn; o.textContent = fn;
          if (fn === curVal) { o.selected = true; matched = true; }
          sel.appendChild(o);
        });
        if (curVal && !matched) {
          const o = document.createElement("option");
          o.value = curVal; o.textContent = curVal + "（未找到）"; o.selected = true;
          sel.appendChild(o);
        }
        sel.onchange = () => { cfg[f.k] = sel.value; markDirty(); };
        wrap2.appendChild(sel);
        row.appendChild(wrap2);
      } else if (f.type === "color") {
        const wrap2 = document.createElement("div");
        wrap2.className = "pf-color";
        const txt = document.createElement("input");
        txt.type = "text"; txt.value = cfg[f.k] || "black";
        const picker = document.createElement("input");
        picker.type = "color"; picker.value = colorHex(cfg[f.k] || "black");
        picker.oninput = () => { txt.value = picker.value; cfg[f.k] = picker.value; markDirty(); };
        txt.oninput = () => { cfg[f.k] = txt.value; markDirty(); };
        wrap2.appendChild(txt); wrap2.appendChild(picker);
        row.appendChild(wrap2);
      } else {
        const inp = document.createElement("input");
        inp.type = "text"; inp.value = cfg[f.k] ?? "";
        inp.oninput = () => {
          cfg[f.k] = (f.type === "int") ? (parseInt(inp.value) || 0) : inp.value;
          markDirty();
        };
        row.appendChild(inp);
      }
      body.appendChild(row);
    });

    head.onclick = () => {
      const closed = body.classList.toggle("closed");
      head.querySelector("span:last-child").textContent = closed ? "▸" : "▾";
    };
    wrap.appendChild(head); wrap.appendChild(body);
    cont.appendChild(wrap);
  });
}

// ── 未保存标记 ───────────────────────────────────────────────
function markDirty() {
  const dot = document.getElementById("dirty-dot");
  if (dot) dot.style.display = "inline";
  // 同时在保存按钮上也标示一下
  const btn = document.getElementById("btn-save");
  if (btn) btn.textContent = "保存 *";
}
function clearDirty() {
  const dot = document.getElementById("dirty-dot");
  if (dot) dot.style.display = "none";
  const btn = document.getElementById("btn-save");
  if (btn) btn.textContent = "保存";
}

function updateStatusBar(cfg) {
  const W = +cfg.canvas_width || 2480, H = +cfg.canvas_height || 1860;
  const cols = +cfg.col_num || 24, rows = +cfg.row_num || 30;
  const lcw = +cfg.leaf_center_width || 120;
  const ml = +cfg.margins_left || 50, mr = +cfg.margins_right || 50;
  const mt = +cfg.margins_top || 200, mb = +cfg.margins_bottom || 50;
  const cw = (W - ml - mr - lcw) / cols;
  const rh = (H - mt - mb) / rows;
  document.getElementById("status-bar").textContent =
    `版式: ${S.currentId}  |  画布 ${W}×${H}  |  ${cols}列×${rows}行  |  列宽 ${cw.toFixed(1)}px  |  行高 ${rh.toFixed(1)}px  |  每页 ${cols*rows} 字`;
}

// ══════════════════════════════════════════════════════════════
//  YAML 序列化
// ══════════════════════════════════════════════════════════════
function cfgToYAML(cfg, id) {
  // YAML 格式: "key:" 后对齐到 26 列（与原 layouts/*.yaml 一致）
  const kv = (k, v) => (k + ":").padEnd(26) + (v ?? "");
  const q = (k, v) => (k + ":").padEnd(22) + '"' + (v ?? "") + '"';
  return [
    `# ── 版式配置：${id} ──`,
    kv("canvas_id", id),
    kv("canvas_width",  cfg.canvas_width  || 2480),
    kv("canvas_height", cfg.canvas_height || 1860),
    "",
    kv("margins_top",    cfg.margins_top    || 200),
    kv("margins_bottom", cfg.margins_bottom || 50),
    kv("margins_left",   cfg.margins_left   || 50),
    kv("margins_right",  cfg.margins_right  || 50),
    "",
    kv("col_num",           cfg.col_num           || 24),
    kv("leaf_center_width", cfg.leaf_center_width || 120),
    kv("outline_width",     cfg.outline_width     || 10),
    kv("outline_vmargin",   cfg.outline_vmargin   || 5),
    "",
    kv("row_num", cfg.row_num || 30) + "           # 每列字数",
    "",
    q("spine_title",    cfg.spine_title    || ""),
    q("title_postfix",  cfg.title_postfix  || ""),
    "",
    kv("if_tpcenter",    cfg.if_tpcenter ?? 1),
    kv("title_font_size",  cfg.title_font_size  || 80),
    q("title_font_color", cfg.title_font_color || "black"),
    kv("title_y",          cfg.title_y          || 1200),
    kv("title_ydis",       cfg.title_ydis       || 1.2),
    "",
    kv("pager_font_size",  cfg.pager_font_size  || 35),
    q("pager_font_color", cfg.pager_font_color || "black"),
    kv("pager_y",          cfg.pager_y          || 500),
    "",
    "# ── 字体 ──",
    q("font1", cfg.font1 || "qiji-combo.ttf"),
    q("font2", cfg.font2 || "HanaMinA.ttf"),
    q("font3", cfg.font3 || "HanaMinB.ttf"),
    "",
    q("text_fonts_array",    cfg.text_fonts_array    || "123"),
    q("comment_fonts_array", cfg.comment_fonts_array || "23"),
    "",
    kv("text_font1_size",    cfg.text_font1_size    || 60),
    kv("text_font2_size",    cfg.text_font2_size    || 42),
    kv("text_font3_size",    cfg.text_font3_size    || 42),
    q("text_font_color",     cfg.text_font_color    || "black"),
    "",
    kv("comment_font1_size", cfg.comment_font1_size || 30),
    kv("comment_font2_size", cfg.comment_font2_size || 30),
    kv("comment_font3_size", cfg.comment_font3_size || 30),
    q("comment_font_color",  cfg.comment_font_color || "black"),
    kv("topcmt_font_size",   cfg.topcmt_font_size   || 20),
    q("topcmt_color",        cfg.topcmt_color       || "red"),
    "",
    "# ── 标点规则 ──",
    q("exp_replace_comma",  ""),
    q("exp_replace_number", "1一|2二|3三|4四|5五|6六|7七|8八|9九|0〇"),
    q("exp_delete_comma",   "．|　"),
    kv("if_nocomma",  cfg.if_nocomma  ?? 0),
    kv("if_onlyperiod", cfg.if_onlyperiod ?? 1),
    q("exp_onlyperiod", "、|，|。|：|；|！|？|〔|〕|「|」|『|』"),
    q("onlyperiod_color", cfg.onlyperiod_color || "red"),
    q("text_comma_nop",  "、|，|。|：|；|！|？"),
    kv("text_comma_nop_size",  1.2),
    kv("text_comma_nop_x",     0.5),
    kv("text_comma_nop_y",     0.2),
    q("text_comma_90",  "「」『』〔〕…"),
    kv("text_comma_90_size",   0.8),
    kv("text_comma_90_x",      0.35),
    kv("text_comma_90_y",      0.6),
    q("comment_comma_nop",  "、|，|。|：|；|！|？"),
    kv("comment_comma_nop_size", 0.7),
    kv("comment_comma_nop_x",    0.8),
    kv("comment_comma_nop_y",    0.1),
    q("comment_comma_90",  "「」『』〔〕…"),
    kv("comment_comma_90_size",  0.8),
    kv("comment_comma_90_x",     0.15),
    kv("comment_comma_90_y",     0.5),
    "",
    "# ── 书名竖线 ──",
    kv("if_book_vline",   cfg.if_book_vline   ?? 1),
    kv("book_line_width", cfg.book_line_width ?? 1),
    q("book_line_color",  cfg.book_line_color || "black"),
    // ── 标注列框/行线（有值才写入）──
    ...(cfg.col_positions ? [
      "",
      "# ── 列框标注（由图片标注工具生成）──",
      kv("col_positions", cfg.col_positions),
      kv("col_widths",    cfg.col_widths || ""),
    ] : []),
    ...(cfg.row_positions ? [
      kv("row_positions", cfg.row_positions),
    ] : []),
    ...(cfg.fish_position ? [
      kv("fish_position", cfg.fish_position),
    ] : []),
    ...(cfg.image_zones ? [
      "",
      "# ── 图片区域标注（由图片标注工具生成）──",
      "# 格式：x,y,w,h;x,y,w,h  （原图像素坐标，可多个区域，分号分隔）",
      kv("image_zones", cfg.image_zones),
    ] : []),
    ...((cfg.no_topcmt !== undefined && cfg.no_topcmt !== "") ? [
      "",
      "# ── 排版控制 ──",
      kv("no_topcmt", cfg.no_topcmt ?? 0) + "              # 1=禁用眉批插入",
    ] : []),
  ].join("\n");
}

// ══════════════════════════════════════════════════════════════
//  运行预览
// ══════════════════════════════════════════════════════════════
async function runPreview() {
  const layout = S.layouts.find(l => l.id === S.currentId);
  if (!layout) { toast("请先选择一个版式"); return; }

  // 静态模式：直接显示预置图
  if (STATIC_MODE) {
    const imgs = STATIC_PREVIEWS[S.currentId];
    if (imgs && imgs.length > 0) {
      showPreviewImages({ images: imgs, pages_total: imgs.length,
        info: { canvas:"预置", cols:"—", rows:"—", col_w_px:"—", row_h_px:"—", chars_per_page:"—" } });
      document.getElementById("status-bar").textContent =
        `${S.currentId}  |  在线演示·预置效果图  |  共 ${imgs.length} 页`;
    } else {
      toast("该版式暂无预置效果图");
    }
    showLocalRunBanner();
    return;
  }

  const pages = +document.getElementById("sel-pages").value || 2;
  const spin  = document.getElementById("preview-spinner");
  const btn1  = document.getElementById("btn-preview");
  const btn2  = document.getElementById("btn-preview2");
  const hint  = document.getElementById("preview-hint");

  // 用当前 cfg 重新生成 yaml（含编辑器里的修改）
  const yaml = cfgToYAML(layout.cfg, layout.id);

  spin.style.display = "inline-block";
  btn1.disabled = btn2.disabled = true;
  hint.textContent = "排版中，请稍候…";

  try {
    const selected = Array.from(S.selectedImages);
    const result = await api("POST", "/api/preview", {
      yaml_content: yaml,
      pages,
      selected_images: selected,
    });

    if (!result.images || result.images.length === 0) {
      toast("排版完成但未生成图片，请检查 layouts/ 和 canvas/ 目录");
      return;
    }

    showPreviewImages(result);
    updateStatusBar(layout.cfg);
    const inf = result.info;
    document.getElementById("status-bar").textContent =
      `${S.currentId}  |  画布 ${inf.canvas}  |  ${inf.cols}列×${inf.rows}行  |  列宽 ${inf.col_w_px}px  |  行高 ${inf.row_h_px}px  |  每页 ${inf.chars_per_page} 字  |  共 ${result.pages_total} 页`;

    toast(`渲染完成：${result.pages_total} 页`);
  } catch (e) {
    toast("排版失败：" + e.message, 5000);
    hint.textContent = "排版失败，查看控制台";
    console.error(e);
  } finally {
    spin.style.display = "none";
    btn1.disabled = btn2.disabled = false;
    hint.textContent = "调用真实排版引擎生成效果图";
  }
}

function showPreviewImages(result) {
  const empty = document.getElementById("preview-empty");
  const pages = document.getElementById("preview-pages");
  empty.style.display = "none";
  pages.style.display = "";
  pages.innerHTML = "";

  result.images.forEach((src, i) => {
    const card = document.createElement("div");
    card.className = "page-card";

    const img = document.createElement("img");
    // 将相对路径转为完整 URL
    img.src = src.startsWith("/") ? API + src : src;
    img.style.width = "auto";
    img.style.maxWidth = "100%";
    // 添加时间戳避免缓存
    img.src += "?t=" + Date.now();
    applyZoomToImg(img, S.zoom);

    const lbl = document.createElement("div");
    lbl.className = "page-label";
    lbl.textContent = `第 ${i + 1} 页`;

    card.appendChild(img);
    card.appendChild(lbl);
    pages.appendChild(card);
  });

  // 保存 imgs 引用以便缩放
  pages._imgs = pages.querySelectorAll("img");
}

function applyZoomToImg(img, zoom) {
  // 等图片加载后设置宽度
  const setW = () => {
    const natural = img.naturalWidth || 2480;
    img.style.width = Math.round(natural * zoom) + "px";
  };
  if (img.naturalWidth) setW();
  else img.onload = setW;
}

function applyZoomToAll() {
  const pages = document.getElementById("preview-pages");
  if (!pages._imgs) return;
  pages._imgs.forEach(img => applyZoomToImg(img, S.zoom));
}

// ══════════════════════════════════════════════════════════════
//  缩放
// ══════════════════════════════════════════════════════════════
function initZoom() {
  document.getElementById("zoom-in").onclick = () => {
    S.zoom = +(S.zoom + 0.1).toFixed(2); setZoomLabel(); applyZoomToAll();
  };
  document.getElementById("zoom-out").onclick = () => {
    S.zoom = Math.max(0.1, +(S.zoom - 0.1).toFixed(2)); setZoomLabel(); applyZoomToAll();
  };
  document.getElementById("zoom-fit").onclick = () => {
    const body = document.getElementById("preview-body");
    const wrap = body.clientWidth - 40;
    const layout = S.layouts.find(l => l.id === S.currentId);
    const w = layout ? (+layout.cfg.canvas_width || 2480) : 2480;
    S.zoom = +(wrap / w).toFixed(3);
    setZoomLabel(); applyZoomToAll();
  };
}
function setZoomLabel() {
  document.getElementById("zoom-val").textContent = Math.round(S.zoom * 100) + "%";
}

// ══════════════════════════════════════════════════════════════
//  新建 / 保存 / 删除
// ══════════════════════════════════════════════════════════════
function initLayoutActions() {
  // 新建：打开弹窗
  document.getElementById("btn-new").onclick = () => {
    if (STATIC_MODE) { showLocalRunBanner(); return; }
    document.getElementById("new-id").value = "";
    renderLayoutList();
    document.getElementById("overlay-new").classList.add("show");
  };

  // 取消
  document.getElementById("btn-new-cancel").onclick = () => {
    document.getElementById("overlay-new").classList.remove("show");
  };

  // 点弹窗背景关闭
  document.getElementById("overlay-new").onclick = (e) => {
    if (e.target === e.currentTarget)
      document.getElementById("overlay-new").classList.remove("show");
  };

  // 确认新建
  document.getElementById("btn-new-confirm").onclick = async () => {
    const id = document.getElementById("new-id").value.trim();
    if (!id || !/^[A-Za-z0-9_-]+$/.test(id)) {
      alert("ID 只能含字母/数字/下划线/连字符"); return;
    }
    if (S.layouts.find(l => l.id === id)) {
      alert("已存在同名版式"); return;
    }
    const baseId = document.getElementById("new-base").value;
    const base = S.layouts.find(l => l.id === baseId);
    const newCfg = base ? JSON.parse(JSON.stringify(base.cfg)) : makeDefaultLayout().cfg;
    newCfg.canvas_id = id;
    const yaml = cfgToYAML(newCfg, id);
    const entry = { id, name: id, yaml, cfg: newCfg };
    S.layouts.push(entry);
    S.currentId = id;
    renderLayoutList();
    selectLayout(id);
    document.getElementById("overlay-new").classList.remove("show");

    try {
      await api("POST", "/api/layout/save", { id, yaml });
      toast(`已创建 layouts/${id}.yaml`);
    } catch {
      toast("（服务器未连接，已在内存中创建，刷新页面会丢失）");
    }
  };

  // 保存
  document.getElementById("btn-save").onclick = async () => {
    if (STATIC_MODE) { showLocalRunBanner(); return; }
    const layout = S.layouts.find(l => l.id === S.currentId);
    if (!layout) return;
    layout.yaml = cfgToYAML(layout.cfg, layout.id);
    try {
      await api("POST", "/api/layout/save", { id: layout.id, yaml: layout.yaml });
      clearDirty();
      toast(`已保存 layouts/${layout.id}.yaml`);
    } catch (e) {
      toast("保存失败：" + e.message);
    }
  };

  // 删除
  document.getElementById("btn-delete").onclick = async () => {
    if (STATIC_MODE) { showLocalRunBanner(); return; }
    if (!S.currentId) return;
    if (!confirm(`确认删除版式 "${S.currentId}"？\n这将删除 layouts/${S.currentId}.yaml`)) return;
    try {
      await api("POST", "/api/layout/delete", { id: S.currentId });
      S.layouts = S.layouts.filter(l => l.id !== S.currentId);
      S.currentId = null;
      renderLayoutList();
      if (S.layouts.length > 0) selectLayout(S.layouts[0].id);
      toast("已删除");
    } catch (e) {
      toast("删除失败：" + e.message);
    }
  };
}

// ══════════════════════════════════════════════════════════════
//  图片素材选择（用于图文排版 image_zones）
// ══════════════════════════════════════════════════════════════
async function loadImageGallery() {
  try {
    const images = await api("GET", "/api/images");
    S._imageList = images || [];
    renderImageGallery();
  } catch (e) {
    console.error("加载图片素材失败:", e);
  }
}

function renderImageGallery() {
  const wrap = document.getElementById("image-gallery");
  const images = S._imageList || [];
  if (!wrap) return;

  if (images.length === 0) {
    wrap.innerHTML = `<div class="empty-tip">暂无图片素材<br>将图片放入 studio/image/ 目录后刷新</div>`;
    updateImageCount();
    return;
  }

  wrap.innerHTML = "";
  images.forEach(img => {
    const item = document.createElement("div");
    item.className = "ig-item" + (S.selectedImages.has(img.name) ? " selected" : "");
    item.dataset.name = img.name;

    const thumb = document.createElement("img");
    thumb.src = img.url.startsWith("/") ? API + img.url : img.url;
    thumb.loading = "lazy";
    thumb.title = img.name;
    thumb.onerror = () => { thumb.style.display = "none"; };

    const label = document.createElement("div");
    label.className = "ig-label";
    label.textContent = img.name;

    item.appendChild(thumb);
    item.appendChild(label);
    item.onclick = () => {
      item.classList.toggle("selected");
      if (S.selectedImages.has(img.name)) {
        S.selectedImages.delete(img.name);
      } else {
        S.selectedImages.add(img.name);
      }
      updateImageCount();
    };
    wrap.appendChild(item);
  });
  updateImageCount();
}

function updateImageCount() {
  const el = document.getElementById("img-sel-count");
  if (el) el.textContent = `已选 ${S.selectedImages.size} 张`;
}

function initImageGallery() {
  document.getElementById("btn-img-refresh").onclick = loadImageGallery;
  document.getElementById("btn-img-clear").onclick = () => {
    S.selectedImages.clear();
    document.querySelectorAll(".ig-item.selected").forEach(el => el.classList.remove("selected"));
    updateImageCount();
  };
}

function showImageSection(show) {
  const el = document.getElementById("sb-images");
  if (el) el.style.display = show ? "" : "none";
}

// ══════════════════════════════════════════════════════════════
//  图片标注 Tab
// ══════════════════════════════════════════════════════════════
function initAnnotate() {
  const zone   = document.getElementById("upload-zone");
  const input  = document.getElementById("upload-input");
  const canvas = document.getElementById("anno-canvas");
  const ctx    = canvas.getContext("2d");

  zone.onclick = () => input.click();
  zone.addEventListener("dragover",  e => { e.preventDefault(); zone.classList.add("over"); });
  zone.addEventListener("dragleave", () => zone.classList.remove("over"));
  zone.addEventListener("drop", e => {
    e.preventDefault(); zone.classList.remove("over");
    const f = e.dataTransfer.files[0];
    if (f?.type.startsWith("image/")) loadImg(f);
  });
  input.onchange = () => { if (input.files[0]) loadImg(input.files[0]); };

  function loadImg(file) {
    S._uploadFile = file;   // 保存原始 File 对象，写入配置时用版式ID重命名上传
    const url = URL.createObjectURL(file);
    S.annoImg = new Image();
    S.annoImg.onload = () => {
      S.annoRects = []; S.annoLines = [];
      document.getElementById("anno-placeholder").style.display = "none";
      canvas.style.display = "block";
      fitCanvas(); drawAnno();
    };
    S.annoImg.src = url;

    // 仅在本地预览，不立即上传——等「写入版式配置」时用版式ID命名再上传
    const baseName = file.name.replace(/\.[^.]+$/, "") || "canvas_img";
    S.annoCanvasId = baseName;

    const statusEl = document.getElementById("upload-status");
    statusEl.style.display = "block";
    statusEl.style.color = "var(--muted)";
    statusEl.textContent = `已加载图片（${S.annoImg?.naturalWidth || "?"}×${S.annoImg?.naturalHeight || "?"}），标注完成后点「写入版式配置」保存`;
    // 等图片加载完再更新尺寸信息
    S.annoImg && S.annoImg.addEventListener("load", () => {
      statusEl.textContent = `已加载图片（${S.annoImg.naturalWidth}×${S.annoImg.naturalHeight}），标注完成后点「写入版式配置」保存`;
    }, { once: true });
  }

  function fitCanvas() {
    const body = document.getElementById("anno-body");
    const maxW = body.clientWidth - 40, maxH = body.clientHeight - 40;
    S.annoDispScale = Math.min(maxW / S.annoImg.naturalWidth, maxH / S.annoImg.naturalHeight, 1);
    const z = S.annoDispScale * S.annoZoom;
    canvas.width  = Math.round(S.annoImg.naturalWidth  * z);
    canvas.height = Math.round(S.annoImg.naturalHeight * z);
  }

  function drawAnno() {
    if (!S.annoImg) return;
    const z = S.annoDispScale * S.annoZoom;
    ctx.clearRect(0, 0, canvas.width, canvas.height);
    ctx.drawImage(S.annoImg, 0, 0, canvas.width, canvas.height);
    const COLORS = {
      body:"rgba(255,110,30,.85)", margin:"rgba(30,100,255,.85)",
      col:"rgba(160,30,210,.85)", row:"rgba(20,170,60,.85)",
      center:"rgba(200,155,0,.85)", col_box:"rgba(160,30,210,.85)",
      hsplit:"rgba(20,170,60,.85)", img_zone:"rgba(220,40,180,.9)"
    };
    const LABELS = {
      body:"正文区域", margin:"眉批区域", col:"列线", row:"行线",
      center:"版心区域", col_box:"列框", hsplit:"分界线", img_zone:"图片区域"
    };
    // 画所有矩形（body/margin/center/col_box/img_zone）
    S.annoRects.forEach((r, idx) => {
      ctx.strokeStyle = COLORS[r.tool]; ctx.lineWidth = 2;
      ctx.strokeRect(r.x*z, r.y*z, r.w*z, r.h*z);
      ctx.fillStyle = COLORS[r.tool].replace(".85","0.1").replace(".9","0.12");
      ctx.fillRect(r.x*z, r.y*z, r.w*z, r.h*z);
      ctx.fillStyle = COLORS[r.tool]; ctx.font = "bold 11px sans-serif"; ctx.textBaseline = "top";
      let label;
      if (r.tool === "col_box")   label = `列${idx+1}`;
      else if (r.tool === "img_zone") {
        const izIdx = S.annoRects.filter((x,i) => x.tool === "img_zone" && i <= idx).length;
        label = `图片区${izIdx}`;
      }
      else label = LABELS[r.tool];
      ctx.fillText(label, r.x*z + 3, r.y*z + 3);
    });
    // 拖拽预览
    if (S.annoDragging && S._dragRect) {
      const r = S._dragRect;
      ctx.strokeStyle = COLORS[S.annoTool] || "rgba(100,100,100,.8)";
      ctx.lineWidth = 1.5; ctx.setLineDash([6,4]);
      ctx.strokeRect(r.x*z, r.y*z, r.w*z, r.h*z); ctx.setLineDash([]);
    }
    // 横切线（眉批/正文分界）
    if (S.annoHSplit !== null) {
      ctx.strokeStyle = COLORS.hsplit; ctx.lineWidth = 2; ctx.setLineDash([10,4]);
      ctx.beginPath();
      ctx.moveTo(0, S.annoHSplit*z); ctx.lineTo(canvas.width, S.annoHSplit*z);
      ctx.stroke(); ctx.setLineDash([]);
      ctx.fillStyle = COLORS.hsplit; ctx.font = "bold 11px sans-serif"; ctx.textBaseline = "top";
      ctx.fillText("眉批/正文分界", 4, S.annoHSplit*z + 3);
    }
    // 旧式列线/行线（兼容）
    S.annoLines.forEach(l => {
      ctx.strokeStyle = COLORS[l.tool] || "#888"; ctx.lineWidth = 1.5; ctx.setLineDash([8,4]);
      ctx.beginPath();
      if (l.tool === "col") { ctx.moveTo(l.x*z,0); ctx.lineTo(l.x*z,canvas.height); }
      else { ctx.moveTo(0,l.y*z); ctx.lineTo(canvas.width,l.y*z); }
      ctx.stroke(); ctx.setLineDash([]);
    });
  }

  canvas.addEventListener("mousedown", e => {
    if (!S.annoImg) return;
    const z = S.annoDispScale * S.annoZoom;
    const r = canvas.getBoundingClientRect();
    const nx = (e.clientX - r.left)/z, ny = (e.clientY - r.top)/z;
    if (S.annoTool === "hsplit") {
      // 横切线：点击即设置 y
      S.annoHSplit = ny;
      updateAnnoResult(); drawAnno();
    } else if (S.annoTool === "row") {
      S.annoLines.push({ tool:"row", x:nx, y:ny });
      updateAnnoResult(); drawAnno();
    } else {
      // body / margin / center / col_box：框选
      S.annoDragging = true; S.annoStart = { x:nx, y:ny };
    }
  });
  canvas.addEventListener("mousemove", e => {
    if (!S.annoImg) return;
    const z = S.annoDispScale * S.annoZoom;
    const r = canvas.getBoundingClientRect();
    const nx = (e.clientX - r.left)/z, ny = (e.clientY - r.top)/z;
    document.getElementById("anno-coord").textContent =
      `显示坐标 (${Math.round(nx)}, ${Math.round(ny)})  ·  原图坐标 (${Math.round(nx*S.annoImg.naturalWidth/canvas.width*z)}, ${Math.round(ny*S.annoImg.naturalHeight/canvas.height*z)})`;
    if (S.annoDragging && S.annoStart) {
      const sx=S.annoStart.x, sy=S.annoStart.y;
      S._dragRect = { x:Math.min(sx,nx), y:Math.min(sy,ny), w:Math.abs(nx-sx), h:Math.abs(ny-sy) };
      drawAnno();
    }
  });
  canvas.addEventListener("mouseup", () => {
    if (!S.annoDragging||!S._dragRect) return;
    if (S._dragRect.w>5 && S._dragRect.h>5) { S.annoRects.push({...S._dragRect, tool:S.annoTool}); updateAnnoResult(); }
    S.annoDragging=false; S._dragRect=null; drawAnno();
  });

  document.querySelectorAll(".anno-btn").forEach(b => {
    b.onclick = () => {
      document.querySelectorAll(".anno-btn").forEach(x=>x.classList.remove("active"));
      b.classList.add("active"); S.annoTool = b.dataset.tool;
    };
  });
  document.getElementById("anno-zoom-in").onclick = () => {
    S.annoZoom = +(S.annoZoom+0.2).toFixed(1);
    document.getElementById("anno-zoom-val").textContent = Math.round(S.annoZoom*100)+"%";
    fitCanvas(); drawAnno();
  };
  document.getElementById("anno-zoom-out").onclick = () => {
    S.annoZoom = Math.max(0.2, +(S.annoZoom-0.2).toFixed(1));
    document.getElementById("anno-zoom-val").textContent = Math.round(S.annoZoom*100)+"%";
    fitCanvas(); drawAnno();
  };
  document.getElementById("btn-anno-clear").onclick = () => {
    S.annoRects=[]; S.annoLines=[]; S.annoHSplit=null; updateAnnoResult(); drawAnno();
  };

  // 打开「选择写入版式」弹窗
  document.getElementById("btn-anno-apply").onclick = () => {
    if (S.annoRects.length === 0 && S.annoLines.length === 0 && S.annoHSplit === null) {
      toast("请先标注至少一个区域"); return;
    }
    openApplyModal();
  };
}

// ── 选择写入版式弹窗 ──────────────────────────────────────────
function openApplyModal() {
  const derived = deriveFromAnnotation();
  const overlay = document.getElementById("overlay-apply");

  // 填充版式选择下拉
  const sel = document.getElementById("apply-target");
  sel.innerHTML = "";
  S.layouts.forEach(l => {
    const o = document.createElement("option");
    o.value = l.id;
    o.textContent = l.id + `  (${l.cfg.col_num||"?"}列×${l.cfg.row_num||"?"}行)`;
    if (l.id === S.currentId) o.selected = true;
    sel.appendChild(o);
  });

  // 预览将写入的参数
  function refreshPreview() {
    const items = Object.entries(derived)
      .map(([k, v]) => `<b>${k}</b> = ${v}`)
      .join("<br>");
    document.getElementById("apply-preview").innerHTML =
      `<div style="color:var(--ink);margin-bottom:4px">将写入以下参数：</div>` + items;
  }
  refreshPreview();
  sel.onchange = refreshPreview;

  overlay.classList.add("show");

  document.getElementById("btn-apply-cancel").onclick = () => {
    overlay.classList.remove("show");
  };
  overlay.onclick = e => { if (e.target === overlay) overlay.classList.remove("show"); };

  document.getElementById("btn-apply-confirm").onclick = async () => {
    const targetId = sel.value;
    const layout = S.layouts.find(l => l.id === targetId);
    if (!layout) { toast("找不到版式：" + targetId); return; }

    // 如果有上传的图片，重命名为版式ID保存到 canvas/
    if (S.annoImg && S._uploadFile) {
      const statusEl = document.getElementById("upload-status");
      statusEl.style.display = "block";
      statusEl.textContent = `正在将背景图保存为 canvas/${targetId}…`;

      try {
        const form = new FormData();
        form.append("file", S._uploadFile, S._uploadFile.name);
        form.append("filename", targetId);   // ← 用版式ID命名
        const r = await fetch(API + "/api/upload_canvas", { method: "POST", body: form });
        const d = await r.json();
        if (d.ok) {
          S.annoCanvasId = d.canvas_id;   // = targetId
          derived.canvas_id = d.canvas_id;
          statusEl.innerHTML = `✓ 背景图已保存为 <b>canvas/${d.filename}</b>`;
          statusEl.style.color = "var(--success, #3a6e3a)";
        } else {
          statusEl.textContent = "图片保存失败：" + (d.error || "");
          statusEl.style.color = "var(--danger)";
        }
      } catch (e) {
        console.error("upload rename error:", e);
      }
    }

    Object.assign(layout.cfg, derived);
    // 立即生成 yaml 并保存到服务器，不依赖用户手动点保存
    layout.yaml = cfgToYAML(layout.cfg, layout.id);
    try {
      await api("POST", "/api/layout/save", { id: layout.id, yaml: layout.yaml });
      selectLayout(targetId);
      overlay.classList.remove("show");
      document.querySelector("[data-tab='preview']").click();
      toast(`已写入并保存版式「${targetId}」`);
    } catch (e) {
      selectLayout(targetId);
      markDirty();
      overlay.classList.remove("show");
      document.querySelector("[data-tab='preview']").click();
      toast(`已写入版式「${targetId}」，但保存失败：${e.message}，请手动点保存`);
    }
  };
}

function updateAnnoResult() {
  const el = document.getElementById("anno-result");
  const btn = document.getElementById("btn-anno-apply");
  const hasAnything = S.annoRects.length || S.annoLines.length || S.annoHSplit !== null;
  if (!hasAnything) {
    el.innerHTML = '<span style="color:var(--muted)">尚未标注</span>';
    btn.disabled = true; return;
  }
  btn.disabled = false;

  const body    = S.annoRects.find(r=>r.tool==="body");
  const marg    = S.annoRects.find(r=>r.tool==="margin");
  const leaf    = S.annoRects.find(r=>r.tool==="center");
  const colBoxes = S.annoRects.filter(r=>r.tool==="col_box");
  const rowLines = S.annoLines.filter(l=>l.tool==="row");
  const imgZones = S.annoRects.filter(r=>r.tool==="img_zone");

  let html = "";
  if (body)  html += `<div><span class="anno-key">正文区域</span> (${Math.round(body.x)},${Math.round(body.y)}) ${Math.round(body.w)}×${Math.round(body.h)}</div>`;
  if (marg)  html += `<div><span class="anno-key">眉批区域</span> (${Math.round(marg.x)},${Math.round(marg.y)}) ${Math.round(marg.w)}×${Math.round(marg.h)}</div>`;
  if (S.annoHSplit !== null) html += `<div><span class="anno-key">分界线</span> y=${Math.round(S.annoHSplit)}</div>`;
  if (leaf)  html += `<div><span class="anno-key">版心区域</span> 宽=${Math.round(leaf.w)}</div>`;
  if (colBoxes.length) html += `<div><span class="anno-key">列框</span> ×${colBoxes.length} → col_num=${colBoxes.length}</div>`;
  if (rowLines.length) html += `<div><span class="anno-key">行线</span> ×${rowLines.length} → row_num=${rowLines.length+1}</div>`;
  if (imgZones.length) {
    imgZones.forEach((r, i) => {
      html += `<div><span class="anno-key" style="color:rgba(220,40,180,.9)">图片区${i+1}</span> (${Math.round(r.x)},${Math.round(r.y)}) ${Math.round(r.w)}×${Math.round(r.h)}</div>`;
    });
  }

  // 自适应推算预览
  const d = deriveFromAnnotation();
  html += `<div style="margin-top:6px;color:var(--muted);font-size:11px">── 推算结果 ──</div>`;
  if (d.col_num)          html += `<div><span class="anno-key">列数</span> col_num=${d.col_num}</div>`;
  if (d.row_num)          html += `<div><span class="anno-key">行数</span> row_num=${d.row_num}</div>`;
  if (d.text_font1_size)  html += `<div><span class="anno-key">字号</span> text_font1_size=${d.text_font1_size}</div>`;
  if (d.leaf_center_width) html += `<div><span class="anno-key">版心宽</span> leaf_center_width=${d.leaf_center_width}</div>`;
  if (d.image_zones)      html += `<div><span class="anno-key" style="color:rgba(220,40,180,.9)">图片区域</span> ${d.image_zones.split(";").length} 个</div>`;
  html += `<div><span class="anno-key">禁用眉批</span> no_topcmt=${d.no_topcmt} ${d.no_topcmt ? "（无眉批标注，自动禁用）" : "（有眉批标注，保留）"}</div>`;

  el.innerHTML = html;
}

function deriveFromAnnotation() {
  const body    = S.annoRects.find(r=>r.tool==="body");
  const marg    = S.annoRects.find(r=>r.tool==="margin");
  const leaf    = S.annoRects.find(r=>r.tool==="center");
  const colBoxes = S.annoRects.filter(r=>r.tool==="col_box").sort((a,b)=>b.x-a.x); // 右→左
  const rowLines = S.annoLines.filter(l=>l.tool==="row").sort((a,b)=>a.y-b.y);
  const d = {};

  // 画布尺寸
  if (S.annoImg) {
    d.canvas_width  = S.annoImg.naturalWidth;
    d.canvas_height = S.annoImg.naturalHeight;
  }
  if (S.annoCanvasId) d.canvas_id = S.annoCanvasId;

  // 正文区域 → margins（优先用正文框，其次用横切线推算眉批底边）
  if (body) {
    d.margins_top    = Math.round(body.y);
    d.margins_bottom = Math.round((d.canvas_height||1860) - body.y - body.h);
    d.margins_left   = Math.round(body.x);
    d.margins_right  = Math.round((d.canvas_width||2480) - body.x - body.w);
  }

  // 眉批区域
  if (marg) {
    d.fish_top_y = Math.round(marg.y + marg.h);
    d.fish_position = marg.y < (body ? body.y : (d.canvas_height||1860)/2) ? "top" : "bottom";
  } else if (S.annoHSplit !== null && body) {
    // 用横切线推算眉批/正文分界
    if (S.annoHSplit < body.y + body.h / 2) {
      // 分界线在正文区域上半 → 眉批在上
      d.fish_top_y    = Math.round(S.annoHSplit);
      d.fish_position = "top";
      d.margins_top   = Math.round(body.y); // 正文从body.y开始
    } else {
      d.fish_top_y    = Math.round(S.annoHSplit);
      d.fish_position = "bottom";
    }
  }

  // 版心宽
  if (leaf) {
    d.leaf_center_width = Math.max(10, Math.round(leaf.w));
  }

  // 列框 → col_positions + col_widths + col_num
  if (colBoxes.length) {
    d.col_num       = colBoxes.length;
    d.col_positions = colBoxes.map(r => Math.round(r.x)).join(",");
    d.col_widths    = colBoxes.map(r => Math.round(r.w)).join(",");

    // 自适应字号：用最窄列宽 × 0.85
    const minW = Math.min(...colBoxes.map(r=>r.w));
    const fontSize = Math.max(10, Math.floor(minW * 0.85 / 2) * 2);
    d.text_font1_size    = fontSize;
    d.text_font2_size    = Math.round(fontSize * 0.7);
    d.text_font3_size    = Math.round(fontSize * 0.7);
    d.comment_font1_size = Math.round(fontSize * 0.5);
    d.comment_font2_size = Math.round(fontSize * 0.5);
    d.comment_font3_size = Math.round(fontSize * 0.5);

    // 自适应行数（无行线时）
    if (!rowLines.length && body) {
      const rh = fontSize * 1.1;
      d.row_num = Math.max(1, Math.floor(body.h / rh));
    }
  } else {
    // 无列框：沿用旧列线逻辑（逐条点击的 col 线）
    const oldCols = S.annoLines.filter(l=>l.tool==="col");
    if (oldCols.length) d.col_num = oldCols.length + 1;
  }

  // 行线
  if (rowLines.length) {
    d.row_num       = rowLines.length + 1;
    d.row_positions = rowLines.map(l => Math.round(l.y)).join(",");
  }

  // 图片区域 → image_zones: "x,y,w,h;x,y,w,h;..."（原图像素坐标）
  const imgZones2 = S.annoRects.filter(r => r.tool === "img_zone");
  if (imgZones2.length) {
    d.image_zones = imgZones2
      .map(r => `${Math.round(r.x)},${Math.round(r.y)},${Math.round(r.w)},${Math.round(r.h)}`)
      .join(";");
  }

  // 无眉批区域标注（无 margin 框 且 无 hsplit 线）→ 自动设 no_topcmt=1
  const hasMeipiAnno = !!marg || S.annoHSplit !== null;
  d.no_topcmt = hasMeipiAnno ? 0 : 1;

  return d;
}

// ══════════════════════════════════════════════════════════════
//  Tab 切换
// ══════════════════════════════════════════════════════════════
function initTabs() {
  document.querySelectorAll(".tab").forEach(btn => {
    btn.onclick = () => {
      document.querySelectorAll(".tab").forEach(b=>b.classList.remove("active"));
      document.querySelectorAll(".tab-panel").forEach(p=>p.classList.remove("active"));
      btn.classList.add("active");
      document.getElementById("tab-"+btn.dataset.tab).classList.add("active");
    };
  });
}

// ══════════════════════════════════════════════════════════════
//  颜色工具
// ══════════════════════════════════════════════════════════════
function colorHex(s) {
  const m = { black:"#1a1208", white:"#ffffff", red:"#c0392b", blue:"#0e6696", gray:"#888888", brown:"#6b4226" };
  return m[s?.toLowerCase()] || (s?.startsWith("#") ? s : "#888888");
}

// ══════════════════════════════════════════════════════════════
//  启动
// ══════════════════════════════════════════════════════════════
async function init() {
  initTabs();
  initZoom();
  initLayoutActions();
  initImageGallery();
  initAnnotate();
  // 不再有 initAdvanced()

  await checkServer();

  // 加载字体列表（供字体下拉使用）
  try {
    S_FONTS = await api("GET", "/api/fonts");
  } catch {
    S_FONTS = ["qiji-combo.ttf", "HanaMinA.ttf", "HanaMinB.ttf", "WenYue_GuTiFangSong_F.otf"];
  }

  // 加载图片素材
  loadImageGallery();
  showImageSection(true);

  await loadLayouts();

  // 预览按钮
  document.getElementById("btn-preview").onclick  = runPreview;
  document.getElementById("btn-preview2").onclick = runPreview;

  // 初始适应缩放
  setTimeout(() => {
    const body = document.getElementById("preview-body");
    const wrap = body.clientWidth - 40;
    const layout = S.layouts.find(l=>l.id===S.currentId);
    const w = layout ? (+layout.cfg.canvas_width||2480) : 2480;
    S.zoom = +(wrap/w).toFixed(3);
    setZoomLabel();
  }, 50);
}

document.addEventListener("DOMContentLoaded", init);
