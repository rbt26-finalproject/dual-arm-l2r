# Laporan Progress 1
## Final Project

## Identitas

- **Anggota 1**: Muhammad Wendy Fyfo Anggara / 2306223906
- **Anggota 2**: Adam Ghaviyasha / 2006482584
- **Tanggal Pengumpulan Progress 1**: 10 Mei
- **Periode Project**: 7 Mei - 3 Juni
- **Final Presentation**: 5 Juni

## 1. Judul Project

Judul project yang dipilih:

```text
Language-Driven Dual-Arm Manipulation with Dynamic Handoff and Obstacle Handling in MuJoCo Playground
```

## 2. Latar Belakang

Sistem robot manipulator modern semakin dituntut untuk mampu memahami instruksi berbasis bahasa alami dan mengeksekusinya secara otonom dalam lingkungan fisika yang realistis. Salah satu tantangan utama dalam robotika adalah menjembatani kesenjangan antara instruksi tingkat tinggi seperti perintah manusia dalam bahasa sehari-hari, dan eksekusi tingkat rendah berupa trajectory dan kontrol motor pada robot.

Reward function memegang peranan krusial sebagai jembatan semantik antara tujuan yang dinyatakan dalam bahasa dan evaluasi kuantitatif kualitas perilaku robot. Melalui reward, sistem dapat mengukur seberapa dekat robot mendekati goal, apakah obstacle berhasil dihindari atau dimanipulasi, dan apakah trajectory yang dihasilkan cukup smooth untuk dieksekusi di simulator fisika. Tanpa reward yang terstruktur, sistem sulit membedakan perilaku yang baik dari yang buruk dalam ruang konfigurasi yang besar.

Kolaborasi dua robot arm menghadirkan tantangan tambahan yang tidak dijumpai pada sistem single-arm: pembagian workspace, koordinasi antar-arm, dan kebutuhan untuk melakukan handoff objek ketika bagian tertentu dari workspace hanya dapat dijangkau oleh salah satu arm. Dalam skenario yang memiliki obstacle, arm pertama mungkin harus memanipulasi obstacle terlebih dahulu sebelum objek target dapat dijangkau dan diserahkan ke arm kedua untuk mencapai goal akhir.

Project ini mengadaptasi framework *Language to Rewards* (Yu et al., 2023) untuk skenario dual-arm dengan workspace terpisah, menggunakan MuJoCo Playground (Zakka et al., 2025) sebagai platform simulasi. Kontribusi utama project adalah pipeline language-to-reward yang mampu mendekomposisi task menjadi per-arm reward function dengan handoff zone yang bersifat dinamis, ditentukan secara otomatis berdasarkan irisan reachable workspace kedua arm dan konfigurasi obstacle pada scene.

## 3. Paper Acuan

### 3.1 Language to Rewards for Robotic Skill Synthesis

Paper *Language to Rewards for Robotic Skill Synthesis* membahas gagasan bahwa instruksi bahasa tingkat tinggi dapat diterjemahkan menjadi reward function untuk menghasilkan perilaku robot. Alur konseptualnya adalah:

```text
Instruction / Goal
-> Motion Description
-> Reward / Cost Function
-> Robot Behavior
```

Dalam project ini, konsep tersebut diadaptasi untuk skenario dual-arm. Pipeline language-to-reward diperluas agar mampu mendekomposisi satu task tunggal menjadi sub-task per arm dengan reward terpisah untuk setiap fase: manipulasi obstacle, handoff, dan penempatan objek di goal. Alur konseptual yang diadaptasi:

```
Instruksi Bahasa  →  Motion Descriptor  →  Task Decomposer
  →  Per-Arm Reward Generator  →  Dynamic Handoff Constraint
  →  MuJoCo Playground Execution
```

### 3.2 Demonstrating MuJoCo Playground

Zakka et al. (2025) memperkenalkan MuJoCo Playground sebagai framework open-source untuk robot learning berbasis MJX/JAX yang memungkinkan training policy dalam hitungan menit pada single GPU. Platform ini menyediakan environment manipulation termasuk bi-arm task dengan robot Aloha, serta mendukung simulasi obstacle dan domain randomization.

Dalam project ini, MuJoCo Playground digunakan sebagai platform simulasi utama. Scene dual-arm dengan workspace terpisah dibangun di atas environment yang tersedia di Playground, dengan ekstensi berupa konfigurasi obstacle dan split-zone workspace yang memaksa kedua arm untuk berkolaborasi melalui mekanisme handoff. Playground juga menyediakan infrastruktur untuk logging reward, rendering video demo, dan evaluasi metrik.

---

## 4. Topik Project yang Dipilih

- **Topik yang dipilih**: Simulasi dual robot arm dengan kolaborasi berbasis reward untuk task obstacle handling dan object handoff di MuJoCo Playground.

- **Alasan memilih topik**:

- Relevansi langsung dengan paper acuan: skenario dual-arm dengan language-to-reward pipeline merupakan ekstensi natural dari *Language to Rewards* (Yu et al., 2023) ke setting multi-agent.
- Novelty dari task: penentuan handoff zone secara dinamis berdasarkan irisan reachable workspace belum dibahas secara eksplisit dalam paper acuan maupun literatur terkait yang ditemukan (RoCo, RoboPARA, Efficient Bimanual Handover).
- Feasibility dalam rentang waktu project: MuJoCo Playground menyediakan environment Aloha bimanual sebagai titik awal, sehingga scene project tidak dibangun dari nol.
- Scope yang terdefinisi jelas: pipeline memiliki input (instruksi bahasa) dan output (eksekusi di simulator) yang konkret, sehingga evaluasi kuantitatif dapat dilakukan.

## 5. Scene dan Robot yang Digunakan

- **Scene XML**: `assets/scene_dual_arm.xml` (dibangun di atas Aloha environment dari MuJoCo Playground)
- **Robot**: Dual-arm Aloha (2x robot arm, konfigurasi meja terpisah)
- **Fokus utama scene**: Obstacle handling + object handoff + split-zone workspace

**Deskripsi scene:**

- Dua robot arm ditempatkan pada dua sisi meja dengan workspace yang sebagian overlap dan sebagian hanya dapat dijangkau oleh satu arm.
- Terdapat satu atau lebih obstacle yang menghalangi objek target. Arm A bertugas memanipulasi obstacle dan menjangkau objek, kemudian menempatkan objek di zona handoff.
- Zona handoff ditentukan secara dinamis sebagai irisan reachable workspace kedua arm, bukan koordinat tetap.
- Arm B mengambil objek dari zona handoff dan menempatkannya di goal position yang hanya dapat dijangkau oleh Arm B.

- *Alasan memilih scene*: scene menciptakan kebutuhan struktural untuk kolaborasi. Tanpa handoff, task tidak dapat diselesaikan oleh satu arm saja, sehingga kolaborasi merupakan requirement fungsional.

## 6. Arsitektur Sistem

```
┌─────────────────────────────────────────────────┐
│              LANGUAGE INSTRUCTION               │
│   e.g. "move the eraser to the right shelf"    │
└────────────────────┬────────────────────────────┘
                     ▼
┌─────────────────────────────────────────────────┐
│              MOTION DESCRIPTOR                  │
│  Decomposes task into per-arm subtask sequence  │
│  + identifies which arm handles which phase     │
└────────────────────┬────────────────────────────┘
                     ▼
┌─────────────────────────────────────────────────┐
│          DYNAMIC HANDOFF ZONE RESOLVER          │
│  Computes intersection of both arms' reachable  │
│  workspaces given current scene & obstacles     │
└────────────────────┬────────────────────────────┘
                     ▼
┌─────────────────────────────────────────────────┐
│           PER-ARM REWARD GENERATOR              │
│  Phase 1 (Arm A): clear obstacle → reach obj   │
│  Phase 2 (Arm A): place obj at handoff zone    │
│  Phase 3 (Arm B): pick from handoff → goal     │
└────────────────────┬────────────────────────────┘
                     ▼
┌─────────────────────────────────────────────────┐
│          MUJOCO PLAYGROUND EXECUTION            │
│  Aloha dual-arm env + MJX/JAX-based training   │
└────────────────────┬────────────────────────────┘
                     ▼
┌─────────────────────────────────────────────────┐
│           METRICS AND VISUALIZATION             │
│  Reward curves, trajectory plots, video demo   │
└─────────────────────────────────────────────────┘
```

## 7. Rencana Implementasi

### `src/main.py`
Entry point untuk menjalankan pipeline utama. Menerima argumen konfigurasi dan instruksi bahasa, kemudian mengorkestrasi seluruh modul dari motion descriptor hingga eksekusi dan visualisasi.

### `src/motion_descriptor.py`
Menerjemahkan instruksi bahasa atau konfigurasi task menjadi motion descriptor terstruktur. Mengidentifikasi objek target, obstacle, posisi goal, dan menentukan alokasi sub-task ke masing-masing arm. Pada tahap awal menggunakan rule-based parser; dapat diperluas ke LLM-based pada iterasi berikutnya.

### `src/handoff_resolver.py`
Menghitung zona handoff secara dinamis berdasarkan reachable workspace kedua arm dan konfigurasi obstacle pada scene. Menghasilkan koordinat atau region handoff yang valid sebagai constraint bagi reward generator.

### `src/reward_function.py`
Mendefinisikan per-arm reward function untuk setiap fase task. Termasuk reward untuk obstacle clearance, reach, handoff placement, dan goal placement. Reward bersifat dense dan dapat dikonfigurasi melalui parameter yang dihasilkan oleh motion descriptor.

### `src/mujoco_runner.py`
Menjalankan training dan eksekusi di MuJoCo Playground. Mengintegrasikan environment Aloha bimanual dengan reward function yang telah dihasilkan, serta mencatat metrik training per episode.

### `src/visualizer.py`
Menghasilkan plot trajectory, reward curve per fase, dan video demo eksekusi. Mendukung analisis perbandingan antar konfigurasi task.

### `configs/default.yaml`
File konfigurasi untuk mendefinisikan scene, task, parameter reward, dan hyperparameter training tanpa mengubah kode sumber.

## 8. Pembagian Kerja Tim

### Anggota 1 — Language, Reward & Planning

- Membaca dan menganalisis paper acuan utama dan literatur terkait.
- Mendesain motion descriptor dan task decomposer.
- Mengimplementasikan handoff zone resolver.
- Membuat per-arm reward function untuk setiap fase.
- Membuat visualisasi reward curve dan trajectory plot.
- Menulis bagian metode dan arsitektur sistem pada laporan.

### Anggota 2 — MuJoCo Playground & Scene

- Mempelajari MuJoCo Playground dan environment Aloha bimanual.
- Menyiapkan dan mengkustomisasi scene XML dual-arm dengan obstacle dan split-zone workspace.
- Mengimplementasikan MuJoCo runner dan integrasi dengan reward function.
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

## 9. Timeline Project

Project dilakukan dari 7 Mei sampai 3 Juni. Final presentation dilakukan pada 5 Juni.

### Week 1 — Progress 1 (7–10 Mei)

Yang dikerjakan/dipersiapkan:

- Penyusunan requirements dan scope project.
- Pemilihan paper acuan dan topik project.
- Perancangan arsitektur sistem.
- Perencanaan scene dan robot.
- Setup repository dan struktur folder.
- Tinjauan literatur terkait.
- Penulisan Laporan Progress 1.

### Week 2 — Progress 2 (11–17 Mei)

Target implementasi awal:

- [x] Scene MuJoCo Playground berjalan dan robot Aloha terlihat di viewer.
- [ ] Split-zone workspace terdefinisi dan obstacle ditempatkan di scene.
- [ ] Motion descriptor rule-based menghasilkan task decomposition.
- [ ] Reward function fase 1 (Arm A: obstacle clearance) dibuat.
- [ ] Trajectory awal Arm A dapat dijalankan di simulator.

### Week 3 — Progress 3 (18–24 Mei)

Target:

- [ ] Handoff zone resolver menghasilkan zona handoff dinamis.
- [ ] Reward function fase 2 dan fase 3 (handoff + Arm B goal placement).
- [ ] Training per-arm reward di MuJoCo Playground berjalan end-to-end.
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
- Scene MuJoCo Playground dan konfigurasi dual-arm.
- Reward function per fase dan hasil training.
- Demo eksekusi simulasi dan analisis metrik.

---

## 10. Rencana Metrik Evaluasi

| Metrik | Deskripsi | Target Awal |
|---|---|---|
| Task success rate | Persentase episode di mana seluruh fase (obstacle clearance, handoff, goal placement) berhasil diselesaikan | ≥ 70% |
| Handoff success rate | Persentase episode di mana objek berhasil dipindahkan dari Arm A ke zona handoff dan diambil Arm B | ≥ 80% |
| Collision count | Jumlah collision antara arm, objek, dan obstacle selama eksekusi | 0 per episode |
| Final distance to goal | Jarak objek terhadap goal position pada akhir episode | < 0.05 m |
| Handoff zone validity | Apakah zona handoff yang dihasilkan benar-benar dapat dijangkau oleh kedua arm | 100% valid |
| Reward curve convergence | Perubahan reward agregat selama training menunjukkan tren meningkat | Monoton meningkat dalam 50k steps |
| Planning time | Waktu yang dibutuhkan untuk menghasilkan task decomposition dan reward function | < 3 detik |

---

## 11. Cara Menjalankan Project

Instalasi dependensi:

```bash
pip install -r requirements.txt
pip install playground
python -c "import playground; print('MuJoCo Playground OK')"
```

Menjalankan pipeline utama:

```bash
python src/main.py --config configs/default.yaml \
                   --task "move the eraser to the right shelf"
```

Training reward di MuJoCo Playground:

```bash
python src/mujoco_runner.py --scene assets/scene_dual_arm.xml \
                            --reward outputs/reward_config.json
```

Visualisasi hasil:

```bash
python src/visualizer.py --metrics outputs/metrics.csv \
                          --reward outputs/reward_log.csv
```

---

## 12. Referensi

### Paper Acuan Utama

1. Wenhao Yu, Nimrod Gileadi, Chuyuan Fu, Sean Kirmani, Kuang-Huei Lee, Montse Gonzalez Arenas, Hao-Tien Lewis Chiang, Tom Erez, Leonard Hasenclever, Jan Humplik, Brian Ichter, Ted Xiao, Peng Xu, Andy Zeng, Tingnan Zhang, Nicolas Heess, Dorsa Sadigh, Jie Tan, Yuval Tassa, Fei Xia. *Language to Rewards for Robotic Skill Synthesis*. CoRL 2023.
2. Kevin Zakka et al. *Demonstrating MuJoCo Playground*. RSS 2025.

### Literatur Terkait

3. Zhao Mandi, Shreeya Jain, Shuran Song. *RoCo: Dialectic Multi-Robot Collaboration with Large Language Models*. ICRA 2024.
4. Yecheng Jason Ma et al. *Eureka: Human-Level Reward Design via Coding Large Language Models*. ICLR 2024.
5. Wanyu Wu et al. *Efficient Bimanual Handover and Rearrangement via Symmetry-Aware Actor-Critic*. ICRA 2023.
6. Shiying Duan et al. *RoboPARA: Dual-Arm Robot Planning with Parallel Allocation and Recomposition Across Tasks*. arXiv 2025.
7. Wenlong Huang et al. *Text2Reward: Reward Shaping with Language Models for Reinforcement Learning*. ICLR 2024.
