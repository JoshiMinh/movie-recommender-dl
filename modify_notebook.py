import json
import copy

with open('Movie_Recommendation_Pipeline.ipynb', 'r', encoding='utf-8') as f:
    nb = json.load(f)

cells = nb['cells']

# 1. Create Cache Loading Cell
cache_cell = {
   "cell_type": "code",
   "execution_count": None,
   "metadata": {},
   "outputs": [],
   "source": [
    "import os\n",
    "import torch\n",
    "import pandas as pd\n",
    "\n",
    "CACHE_FILE = 'training_cache.pt'\n",
    "SKIP_TRAINING = os.path.exists(CACHE_FILE)\n",
    "\n",
    "if SKIP_TRAINING:\n",
    "    print(\"✅ Phát hiện dữ liệu đã huấn luyện từ trước! Tự động bỏ qua bước train.\")\n",
    "    cache = torch.load(CACHE_FILE, map_location=device)\n",
    "    results_arch = cache['results_arch']\n",
    "    results_opt = cache['results_opt']\n"
   ]
}

# 2. Add the cache cell after cell 3 (index 3 is Load Data code)
cells.insert(4, cache_cell)

# Now architecture comparison code is at index 6
cell6_source = cells[6]['source']
new_cell6_source = ["if not SKIP_TRAINING:\n"]
for line in cell6_source:
    new_cell6_source.append("    " + line)
cells[6]['source'] = new_cell6_source

# Optimizer code is at index 10
cell10_source = cells[10]['source']
new_cell10_source = ["if not SKIP_TRAINING:\n"]
for line in cell10_source:
    new_cell10_source.append("    " + line)
cells[10]['source'] = new_cell10_source

# Append cache saving to Optimizer result cell (index 12)
cells[12]['source'].extend([
    "\n",
    "if not SKIP_TRAINING:\n",
    "    print(\"\\nLưu dữ liệu huấn luyện vào ổ cứng để chạy nhanh lần sau...\")\n",
    "    torch.save({'results_arch': results_arch, 'results_opt': results_opt}, CACHE_FILE)\n"
])

# Interactive Demo Setup: Remove the explicit run at the end of cell 14
cell14_source = cells[14]['source']
for i, line in enumerate(cell14_source):
    if "input_history =" in line or "predict_next_movie" in line and "best_model" in line:
        cell14_source[i] = "# " + line

# Add Interactive Widget Cell at the end
widget_cell = {
   "cell_type": "code",
   "execution_count": None,
   "metadata": {},
   "outputs": [],
   "source": [
    "import ipywidgets as widgets\n",
    "from IPython.display import display, clear_output\n",
    "\n",
    "# Lấy danh sách tên phim từ DataFrame để làm gợi ý tìm kiếm (autocomplete)\n",
    "all_titles = movies_df['title'].dropna().unique().tolist()\n",
    "\n",
    "# 1. Khởi tạo các thành phần giao diện (UI)\n",
    "movie_input = widgets.Combobox(\n",
    "    placeholder='Gõ tên phim (VD: Toy Story)',\n",
    "    options=all_titles,\n",
    "    description='Tìm phim:',\n",
    "    ensure_option=False,\n",
    "    disabled=False,\n",
    "    layout=widgets.Layout(width='400px')\n",
    ")\n",
    "\n",
    "add_button = widgets.Button(description='Thêm vào lịch sử', button_style='info', icon='plus')\n",
    "predict_button = widgets.Button(description='Dự đoán phim mới', button_style='success', icon='magic')\n",
    "clear_button = widgets.Button(description='Làm mới', button_style='warning', icon='refresh')\n",
    "\n",
    "# Các vùng để hiển thị kết quả\n",
    "history_display = widgets.Output()\n",
    "result_display = widgets.Output()\n",
    "\n",
    "current_history = []\n",
    "\n",
    "def add_movie(b):\n",
    "    if movie_input.value:\n",
    "        current_history.append(movie_input.value)\n",
    "        with history_display:\n",
    "            clear_output()\n",
    "            print(f\"🎬 Lịch sử hiện tại: {', '.join(current_history)}\")\n",
    "        movie_input.value = ''\n",
    "\n",
    "def clear_history(b):\n",
    "    current_history.clear()\n",
    "    movie_input.value = ''\n",
    "    with history_display:\n",
    "        clear_output()\n",
    "        print(\"🎬 Lịch sử hiện tại: (Trống)\")\n",
    "    with result_display:\n",
    "        clear_output()\n",
    "\n",
    "def run_prediction(b):\n",
    "    with result_display:\n",
    "        clear_output()\n",
    "        if not current_history:\n",
    "            print(\"⚠️ Vui lòng thêm ít nhất 1 phim vào lịch sử trước khi dự đoán!\")\n",
    "            return\n",
    "        \n",
    "        print(f\"⏳ Đang dự đoán dựa trên {len(current_history)} phim...\")\n",
    "        print(\"-\" * 50)\n",
    "        predict_next_movie(current_history, best_model)\n",
    "\n",
    "add_button.on_click(add_movie)\n",
    "clear_button.on_click(clear_history)\n",
    "predict_button.on_click(run_prediction)\n",
    "\n",
    "with history_display:\n",
    "    print(\"🎬 Lịch sử hiện tại: (Trống)\")\n",
    "\n",
    "buttons_layout = widgets.HBox([add_button, clear_button, predict_button])\n",
    "main_ui = widgets.VBox([\n",
    "    widgets.HTML(\"<h2>🍿 Demo Tương tác: Hệ thống Gợi ý Phim</h2>\"),\n",
    "    widgets.HTML(\"<p><i>1. Nhập tên phim và bấm <b>Thêm vào lịch sử</b> (có gợi ý tự động).<br>2. Bấm <b>Dự đoán phim mới</b> để xem kết quả.</i></p>\"),\n",
    "    movie_input,\n",
    "    buttons_layout,\n",
    "    history_display,\n",
    "    result_display\n",
    "])\n",
    "\n",
    "display(main_ui)\n"
   ]
}

cells.append(widget_cell)

with open('Movie_Recommendation_Pipeline.ipynb', 'w', encoding='utf-8') as f:
    json.dump(nb, f, indent=1)
    
print("Notebook updated successfully!")
