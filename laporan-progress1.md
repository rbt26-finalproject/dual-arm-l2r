# Laporan Progress 1
## Final Project

## Identitas

- **Anggota 1**: Muhammad Wendy Fyfo Anggara / 2306223906
- **Anggota 2**: Adam Ghaviyasha / 2006482584
- **Tanggal Pengumpulan Progress 1**: 10 Mei
- **Periode Project**: 7 Mei - 3 Juni
- **Final Presentation**: 5 Juni
---

## 1. Judul Project

```
Language-Driven Dual-Arm Manipulation with Dynamic Handoff and Obstacle Handling in MuJoCo Playground
```

---

## 2. Latar Belakang

Sistem robot manipulator modern semakin dituntut untuk mampu memahami instruksi berbasis bahasa alami dan mengeksekusinya secara otonom dalam environment simulasi yang realistis. Salah satu tantangan utama dalam robotika adalah menjembatani kesenjangan antara instruksi tingkat tinggi, seperti perintah manusia dalam bahasa sehari-hari dan eksekusi tingkat rendah berupa trajectory dan kontrol motor pada robot.

Reward function memegang peranan krusial sebagai jembatan semantik antara target yang dinyatakan dalam bahasa dan evaluasi kuantitatif kualitas perilaku robot. Melalui reward, sistem dapat mengukur seberapa dekat robot mendekati goal, apakah obstacle berhasil dihindari atau dimanipulasi, dan apakah trajectory yang dihasilkan dapat dieksekusi di simulator fisika. Tanpa reward yang terstruktur, sistem sulit membedakan perilaku yang baik dari yang buruk dalam ruang konfigurasi yang besar.

Kolaborasi dua robot arm menghadirkan tantangan tambahan yang tidak dijumpai pada sistem single-arm: pembagian workspace, koordinasi antar-arm, dan kebutuhan untuk melakukan handoff objek ketika bagian tertentu dari workspace hanya dapat dijangkau oleh salah satu arm. Dalam skenario yang memiliki obstacle, arm pertama mungkin harus memanipulasi obstacle terlebih dahulu sebelum objek target dapat dijangkau dan diserahkan ke arm kedua untuk mencapai goal akhir.

Project ini mengadaptasi framework *Language to Rewards* (Yu et al., 2023) untuk skenario dual-arm dengan workspace terpisah, menggunakan MuJoCo Playground (Zakka et al., 2025) sebagai platform simulasi. LLM digunakan sebagai Motion Descriptor yang menerjemahkan instruksi bahasa menjadi dekomposisi task terstruktur, dan sebagai Reward Coder yang menghasilkan kode reward function yang dapat dieksekusi untuk setiap fase. Kontribusi utama project adalah pipeline language-to-reward untuk setting dual-arm dengan handoff zone yang bersifat dinamis, berbeda dari pendekatan RoCo (Mandi et al., 2024) yang menggunakan LLM untuk menghasilkan waypoint XYZ bagi IK.

---

## 3. Paper Acuan

### 3.1 Language to Rewards for Robotic Skill Synthesis

Yu et al. (2023) memperkenalkan paradigma penggunaan reward function sebagai antarmuka semantik antara instruksi bahasa tingkat tinggi dan perilaku robot tingkat rendah. Sistem terdiri dari dua komponen LLM: Motion Descriptor yang menerjemahkan instruksi pengguna menjadi deskripsi gerak terstruktur, dan Reward Coder yang mengonversi deskripsi tersebut menjadi kode reward function yang dapat dieksekusi. Reward yang dihasilkan kemudian dioptimalkan menggunakan MuJoCo MPC.

Dalam project ini, pipeline tersebut diadaptasi untuk skenario dual-arm menggunakan Gemini sebagai LLM. Motion Descriptor menerima instruksi bahasa natural dan menghasilkan TaskDecomposition berformat JSON yang mendefinisikan urutan fase, alokasi arm, dan dependensi antar fase. Reward Coder kemudian menghasilkan kode Python reward function untuk setiap fase secara terpisah. Alur yang diadaptasi:

```
Instruksi Bahasa
  → [Gemini] Motion Descriptor → TaskDecomposition (JSON)
  → [Gemini] Reward Coder      → Per-phase reward code (Python)
  → MuJoCo Playground Execution
```

Perbedaan utama dari paper asli: pipeline diperluas untuk menangani koordinasi dua arm dengan fase handoff, di mana reward untuk fase Arm A dan Arm B dihasilkan secara terpisah dan dieksekusi secara berurutan berdasarkan dependensi yang didefinisikan dalam TaskDecomposition.

### 3.2 Demonstrating MuJoCo Playground

Zakka et al. (2025) memperkenalkan MuJoCo Playground sebagai framework open-source untuk robot learning berbasis MJX/JAX yang memungkinkan training policy dalam hitungan menit pada single GPU. Platform ini menyediakan environment manipulation termasuk bi-arm task dengan robot Aloha, serta mendukung simulasi obstacle dan domain randomization.

Dalam project ini, MuJoCo Playground digunakan sebagai platform simulasi utama. Scene dual-arm dengan workspace terpisah dibangun di atas environment Aloha yang tersedia di Playground, dengan ekstensi berupa dua side table (table A dan table B) yang masing-masing hanya dapat dijangkau oleh satu arm, serta objek dan obstacle yang diletakkan di atasnya sebagai konfigurasi task.


### 3.3 RoCo: Dialectic Multi-Robot Collaboration with Large Language Models

Mandi et al. (2024) memperkenalkan RoCo, sistem multi-robot collaboration di mana setiap robot dikendalikan oleh LLM agent terpisah yang berkomunikasi satu sama lain melalui dialog bahasa natural. Setiap agent menghasilkan waypoint XYZ untuk robot-nya, yang kemudian dieksekusi menggunakan IK dan RRT motion planner.

Dalam konteks project ini, RoCo menjadi referensi penting untuk memahami pendekatan alternatif koordinasi multi-arm berbasis LLM. Perbedaan mendasar antara pendekatan RoCo dan project ini adalah pada output LLM: RoCo menggunakan LLM untuk menghasilkan waypoint koordinat XYZ yang diumpankan ke IK solver, sedangkan project ini mengikuti L2R (Yu et al., 2023) dan menggunakan LLM untuk menghasilkan kode reward function yang dioptimalkan oleh controller secara otomatis. Pendekatan berbasis reward lebih fleksibel karena controller bebas menemukan trajectory terbaik tanpa dibatasi waypoint diskrit yang ditentukan LLM.

---

## 4. Topik Project yang Dipilih

- **Topik yang dipilih**:  Simulasi dual robot arm dengan kolaborasi berbasis language-to-reward untuk task obstacle handling dan object handoff di MuJoCo Playground, menggunakan LLM (Gemini) sebagai Motion Descriptor dan Reward Coder.

- **Alasan memilih topik:**

- Relevansi dengan paper acuan: pipeline language -> reward -> execution merupakan implementasi langsung dari L2R (Yu et al., 2023), diperluas ke setting dual-arm yang belum dibahas dalam paper asli.
- Diferensiasi dari RoCo: tidak seperti RoCo yang menggunakan LLM untuk menghasilkan waypoint XYZ, project ini mengikuti L2R dengan menggunakan LLM untuk menghasilkan kode reward function, sehingga controller yang menentukan trajectory secara otomatis melalui optimisasi reward.
- Novelty: penentuan handoff zone secara dinamis berdasarkan irisan reachable workspace kedua arm, bukan koordinat tetap yang di-hardcode.
- Feasibility: MuJoCo Playground menyediakan environment Aloha bimanual sebagai titik awal, dan Gemini API menyediakan free tier yang cukup untuk kebutuhan project.

---

## 5. Scene dan Robot yang Digunakan

- **Scene XML**: `assets/scene_dual_arm.xml` (dibangun di atas Aloha environment dari MuJoCo Playground)
- **Robot**: Dual-arm Aloha (2x robot arm, konfigurasi meja terpisah)
- **Fokus utama scene**: Obstacle handling + object handoff + split-zone workspace

**Deskripsi scene:**

- Dua robot arm Aloha menghadap ke tengah di atas meja utama. Di sebelah kiri terdapat table A yang hanya dapat dijangkau Arm A (left), dan di sebelah kanan terdapat table B yang hanya dapat dijangkau Arm B (right).
- Objek target (red box dari Playground handover task) dapat diletakkan di table A atau table B melalui konfigurasi task.
- Zona handoff berada di tengah meja utama, yang merupakan irisan reachable workspace kedua arm.

- **Alasan memilih scene:** workspace yang terpisah menciptakan kebutuhan struktural untuk kolaborasi karena task tidak dapat diselesaikan oleh satu arm saja.

---

## 6. Arsitektur Sistem

```
┌─────────────────────────────────────────────────────────┐
│                  LANGUAGE INSTRUCTION                   │
│   e.g. "move the box behind the cup to table B"         │
└───────────────────────┬─────────────────────────────────┘
                        ▼
┌─────────────────────────────────────────────────────────┐
│         MOTION DESCRIPTOR  [Gemini API call 1]          │
│  Outputs TaskDecomposition (JSON):                      │
│  - object, obstacle, start/goal zone                    │
│  - ordered phases with arm assignment and dependencies  │
└───────────────────────┬─────────────────────────────────┘
                        ▼
┌─────────────────────────────────────────────────────────┐
│         REWARD CODER  [Gemini API call per phase]       │
│  Generates executable Python reward function code       │
│  for each phase. Falls back to hardcoded if unavailable │
└───────────────────────┬─────────────────────────────────┘
                        ▼
┌─────────────────────────────────────────────────────────┐
│              DYNAMIC HANDOFF ZONE RESOLVER              │
│  Computes intersection of both arms' reachable          │
│  workspaces given current scene & obstacle positions    │
└───────────────────────┬─────────────────────────────────┘
                        ▼
┌─────────────────────────────────────────────────────────┐
│               MUJOCO PLAYGROUND EXECUTION               │
│  Aloha dual-arm env, per-phase reward, IK + trajectory  │
└───────────────────────┬─────────────────────────────────┘
                        ▼
┌─────────────────────────────────────────────────────────┐
│              METRICS AND VISUALIZATION                  │
│  Reward curves per phase, trajectory plots, video demo  │
└─────────────────────────────────────────────────────────┘
```

---

## 7. Rencana Implementasi

### `src/motion_descriptor.py`
Memanggil Gemini API dengan instruksi bahasa dan system prompt yang mendefinisikan format TaskDecomposition (JSON). Output berisi urutan fase, alokasi arm per fase, target per fase, dan dependensi antar fase. Tersedia rule-based fallback yang aktif otomatis jika `GEMINI_API_KEY` tidak di-set.

### `src/reward_function.py`
Memanggil Gemini API per fase dengan system prompt yang mendefinisikan signature fungsi reward dan constraint output (Python code only). Mengompilasi kode yang dihasilkan dengan `exec()` dalam namespace yang sudah include `numpy` dan `mujoco`. Tersedia hardcoded fallback per action type jika generasi gagal. Semua fungsi reward dikemas dalam `RewardPipeline` yang menyediakan interface `compute(phase_id, data, ...)`.

### `src/handoff_resolver.py`
Menghitung zona handoff secara dinamis dengan sampling joint space kedua arm dan menghitung convex hull dari EE positions yang valid. Menghasilkan center dan radius zona handoff yang dijamin dapat dijangkau keduanya.

### `src/trajectory_arm_a.py`
Demo eksekusi fase 1: menggunakan `motion_descriptor` dan `reward_function`, menyelesaikan IK dengan `scipy.optimize.minimize`, menginterpolasi joint trajectory dari home ke target, dan menjalankan di MuJoCo viewer dengan reward ditampilkan real-time.

### `src/mujoco_runner.py`
Menjalankan training end-to-end di MuJoCo Playground dengan reward dari pipeline. Logging metrik per episode.

### `src/visualizer.py`
Plot reward curve per fase, trajectory plot, dan export video demo.

### `configs/default.yaml`
Konfigurasi scene, task string, parameter reward, dan hyperparameter training.

### `src/main.py`
Entry point yang mengorkestrasi seluruh pipeline dari instruksi bahasa hingga eksekusi dan visualisasi.

---

## 8. Pembagian Kerja Tim

### Anggota 1 — Language, Reward & Planning

- Membaca dan menganalisis paper Language to Rewards (Yu et al., 2023).
- Mengimplementasikan motion descriptor dengan Gemini API dan rule-based fallback.
- Mengimplementasikan reward coder dengan Gemini API dan hardcoded fallback.
- Mengimplementasikan handoff zone resolver.
- Membuat visualisasi reward curve dan trajectory plot.
- Menulis bagian metode dan arsitektur sistem pada laporan.

### Anggota 2 — MuJoCo Playground & Scene

- Mempelajari MuJoCo Playground dan environment Aloha bimanual.
- Menyiapkan scene XML dual-arm dengan table A, table B, obstacle, dan objek target.
- Mengimplementasikan MuJoCo runner dan trajectory executor.
- Melakukan training dan evaluasi eksperimen.
- Membuat video demo eksekusi.
- Menulis bagian eksperimen dan hasil pada laporan.

### Tanggung Jawab Bersama

- Menentukan topik dan scope final project.
- Mendesain scene dan konfigurasi task.
- Menyusun README dan dokumentasi repository.
- Mengelola pull request dan code review.
- Menjalankan eksperimen akhir dan analisis hasil.
- Menyiapkan slide dan materi final presentation.

---

## 9. Timeline Project

### Week 1 — Progress 1 (7–10 Mei)

Yang telah dikerjakan:

- Penyusunan requirements dan scope project.
- Pemilihan paper acuan dan topik project.
- Perancangan arsitektur sistem.
- Tinjauan literatur terkait.
- Penulisan Laporan Progress 1.

### Week 2 — Progress 2 (11–17 Mei)

Target implementasi awal:

- [x] Scene MuJoCo Playground berjalan dan robot Aloha terlihat di viewer.
- [x] Split-zone workspace terdefinisi (table A dan table B ditambahkan ke scene).
- [x] Motion descriptor dengan Gemini dan rule-based fallback menghasilkan TaskDecomposition.
- [x] Reward pipeline dengan Gemini Reward Coder dan hardcoded fallback dibuat.
- [x] Trajectory awal Arm A (IK + joint interpolation) dapat dijalankan di simulator.

### Week 3 — Progress 3 (18–24 Mei)

Target:

- [X] Handoff zone resolver menghasilkan zona handoff dinamis.
- [ ] Reward function fase 2 dan fase 3 (handoff + Arm B goal placement) berjalan end-to-end.
- [ ] Training per-arm reward di MuJoCo Playground berjalan.
- [ ] Reward log dan trajectory plot tersedia.
- [ ] Video demo progress minggu 3.

### Week 4 — Progress 4 (25 Mei–3 Juni)

Target finalisasi:

- [ ] Eksperimen lengkap dengan evaluasi metrik.
- [ ] Video demo final.
- [ ] Slide presentasi selesai.
- [ ] README diperbarui dengan instruksi reproduksi eksperimen.

### Final Presentation (5 Juni)

Yang dipresentasikan:

- Latar belakang dan motivasi project.
- Tinjauan paper acuan dan literatur terkait.
- Arsitektur sistem dan pipeline language-to-reward.
- Demo Gemini sebagai Motion Descriptor dan Reward Coder.
- Scene MuJoCo Playground dan konfigurasi dual-arm.
- Reward function per fase dan hasil training.
- Demo eksekusi simulasi dan analisis metrik.

---

## 10. Rencana Metrik Evaluasi

| Metrik | Deskripsi | Target |
|---|---|---|
| Task success rate | Persentase episode di mana seluruh fase berhasil diselesaikan | ≥ 70% |
| Handoff success rate | Persentase episode di mana objek berhasil di-handoff dari Arm A ke Arm B | ≥ 80% |
| Collision count | Jumlah collision selama eksekusi | 0 per episode |
| Final distance to goal | Jarak objek terhadap goal position pada akhir episode | < 0.05 m |
| Handoff zone validity | Zona handoff yang dihasilkan dapat dijangkau kedua arm | 100% valid |
| Reward curve convergence | Reward agregat meningkat selama training | Monoton meningkat dalam 50k steps |
| LLM parse success rate | Persentase task string yang berhasil di-parse Gemini menjadi TaskDecomposition valid | ≥ 95% |

---

## 11. Cara Menjalankan Project

Setup awal (satu kali):

```bash
# Clone Aloha assets saja (30MB, bukan seluruh menagerie 1.9GB)
mkdir aloha_menagerie && cd aloha_menagerie
git clone --depth 1 --filter=blob:none --sparse https://github.com/google-deepmind/mujoco_menagerie.git .
git sparse-checkout set aloha
cd ..

# Install dependencies
pip install -U "jax[cuda12]"   # atau: pip install jax (CPU only)
pip install -r requirements.txt

# Set Gemini API key (gratis di aistudio.google.com)
export GEMINI_API_KEY=your_key_here
```

Menjalankan demo:

```bash
# Reach demo: kedua arm sweep horizontal
python reach_demo.py

# Motion descriptor saja (test parse task)
python src/motion_descriptor.py

# Trajectory Arm A fase 1 dengan reward live
python src/trajectory_arm_a.py
```

Pipeline utama (Week 3+):

```bash
python src/main.py --config configs/default.yaml \
                   --task "move the box behind the cup to table B"
```

---

## 12. Referensi

### Paper Acuan Utama

1. Wenhao Yu, Nimrod Gileadi, Chuyuan Fu, Sean Kirmani, Kuang-Huei Lee, Montse Gonzalez Arenas, Hao-Tien Lewis Chiang, Tom Erez, Leonard Hasenclever, Jan Humplik, Brian Ichter, Ted Xiao, Peng Xu, Andy Zeng, Tingnan Zhang, Nicolas Heess, Dorsa Sadigh, Jie Tan, Yuval Tassa, Fei Xia. *Language to Rewards for Robotic Skill Synthesis*. CoRL 2023.
2. Kevin Zakka et al. *Demonstrating MuJoCo Playground*. RSS 2025.
3. Zhao Mandi, Shreeya Jain, Shuran Song. *RoCo: Dialectic Multi-Robot Collaboration with Large Language Models*. ICRA 2024.
