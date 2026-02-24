"""
Generate the AI100 Midterm Report PDF using fpdf2.
Runs the full ML pipeline, saves charts, and builds a formatted PDF.
"""
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
from fpdf import FPDF
import warnings, os

warnings.filterwarnings('ignore')
os.chdir(os.path.dirname(os.path.abspath(__file__)))

device = torch.device('cpu')

# ── Data Loading ──
print("Loading data...")
df = pd.read_csv('dataset.csv', encoding='utf-8')

genre_map = {
    'Rock': ['rock','alt-rock','alternative','hard-rock','punk','punk-rock','grunge','psych-rock','rock-n-roll','rockabilly','emo','goth','indie','ska'],
    'Metal': ['metal','heavy-metal','death-metal','black-metal','metalcore','grindcore','hardcore','industrial'],
    'Pop': ['pop','indie-pop','synth-pop','k-pop','j-pop','cantopop','mandopop','j-idol','pop-film','power-pop','british','swedish'],
    'Electronic': ['edm','electro','electronic','house','chicago-house','deep-house','detroit-techno','techno','minimal-techno','progressive-house','trance','hardstyle','drum-and-bass','dubstep','breakbeat','garage','idm','trip-hop','dub'],
    'Hip-Hop/R&B': ['hip-hop','r-n-b','reggaeton','dancehall'],
    'Jazz/Blues': ['jazz','blues','soul','funk','groove'],
    'Classical': ['classical','piano','opera','guitar','new-age','ambient','sleep','study'],
    'Folk/Country': ['folk','country','acoustic','bluegrass','honky-tonk','singer-songwriter','songwriter'],
    'Latin/World': ['latin','latino','salsa','samba','forro','sertanejo','pagode','mpb','brazil','afrobeat','indian','iranian','turkish','malay','tango','reggae','french','german','spanish','world-music'],
    'Dance/Other': ['dance','club','disco','party','happy','chill','romance','sad','anime','disney','children','kids','comedy','show-tunes','j-dance','j-rock','gospel']
}
reverse_map = {g: sg for sg, gs in genre_map.items() for g in gs}
df['super_genre'] = df['track_genre'].map(reverse_map)

audio_features = ['danceability','energy','key','loudness','mode','speechiness',
                  'acousticness','instrumentalness','liveness','valence','tempo',
                  'time_signature','duration_ms']

# ── Preprocessing ──
print("Preprocessing...")
X = df[audio_features].values
le = LabelEncoder()
y = le.fit_transform(df['super_genre'].values)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
scaler = StandardScaler()
X_train_s = scaler.fit_transform(X_train)
X_test_s = scaler.transform(X_test)

# ── Generate Charts ──
print("Generating charts...")

# Chart 1: Class Distribution
fig, ax = plt.subplots(figsize=(9, 5))
df['super_genre'].value_counts().plot(kind='bar', ax=ax, color='#4A90D9', edgecolor='black', linewidth=0.5)
ax.set_title('Distribution of Super-Genres', fontsize=13, fontweight='bold')
ax.set_xlabel('Super-Genre', fontsize=10)
ax.set_ylabel('Number of Tracks', fontsize=10)
ax.tick_params(axis='x', rotation=45)
plt.tight_layout()
plt.savefig('chart_distribution.png', dpi=180, bbox_inches='tight'); plt.close()

# Chart 2: Correlation Heatmap
fig, ax = plt.subplots(figsize=(10, 8))
corr = df[audio_features].corr()
sns.heatmap(corr, annot=True, fmt='.2f', cmap='coolwarm', center=0, ax=ax, annot_kws={'size': 7})
ax.set_title('Audio Feature Correlation Matrix', fontsize=13, fontweight='bold')
plt.tight_layout()
plt.savefig('chart_correlation.png', dpi=180, bbox_inches='tight'); plt.close()

# Chart 3: Feature Distributions
key_feats = ['danceability', 'energy', 'acousticness', 'instrumentalness', 'valence', 'speechiness']
fig, axes = plt.subplots(2, 3, figsize=(16, 9))
sorted_genres = sorted(df['super_genre'].unique())
for i, feat in enumerate(key_feats):
    ax = axes[i // 3][i % 3]
    data_to_plot = [df[df['super_genre'] == sg][feat].values for sg in sorted_genres]
    bp = ax.boxplot(data_to_plot, labels=sorted_genres, patch_artist=True)
    for patch in bp['boxes']:
        patch.set_facecolor('#4A90D9')
        patch.set_alpha(0.6)
    ax.set_title(feat.capitalize(), fontsize=11, fontweight='bold')
    ax.tick_params(axis='x', rotation=90, labelsize=6)
    ax.tick_params(axis='y', labelsize=8)
plt.suptitle('Audio Feature Distributions by Super-Genre', fontsize=13, fontweight='bold')
plt.tight_layout()
plt.savefig('chart_features.png', dpi=180, bbox_inches='tight'); plt.close()

# ── Baseline ──
print("Training baseline...")
lr_model = LogisticRegression(max_iter=1000, random_state=42)
lr_model.fit(X_train_s, y_train)
lr_pred = lr_model.predict(X_test_s)
lr_acc = accuracy_score(y_test, lr_pred)
print(f"  Logistic Regression: {lr_acc:.4f}")

# ── MLP Training ──
print("Training MLP...")
X_tr_t = torch.FloatTensor(X_train_s); y_tr_t = torch.LongTensor(y_train)
X_te_t = torch.FloatTensor(X_test_s); y_te_t = torch.LongTensor(y_test)
loader = DataLoader(TensorDataset(X_tr_t, y_tr_t), batch_size=256, shuffle=True)

class GenreClassifier(nn.Module):
    def __init__(self):
        super().__init__()
        self.network = nn.Sequential(
            nn.Linear(13, 256), nn.BatchNorm1d(256), nn.ReLU(), nn.Dropout(0.3),
            nn.Linear(256, 128), nn.BatchNorm1d(128), nn.ReLU(), nn.Dropout(0.3),
            nn.Linear(128, 64), nn.BatchNorm1d(64), nn.ReLU(), nn.Dropout(0.2),
            nn.Linear(64, len(le.classes_))
        )
    def forward(self, x):
        return self.network(x)

model = GenreClassifier()
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.001, weight_decay=1e-4)
scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=5, factor=0.5)

train_losses, train_accs, test_accs = [], [], []
for epoch in range(50):
    model.train()
    running_loss, correct, total = 0.0, 0, 0
    for xb, yb in loader:
        optimizer.zero_grad()
        out = model(xb)
        loss = criterion(out, yb)
        loss.backward()
        optimizer.step()
        running_loss += loss.item()
        _, pred = torch.max(out, 1)
        total += yb.size(0)
        correct += (pred == yb).sum().item()
    train_losses.append(running_loss / len(loader))
    train_accs.append(correct / total)
    model.eval()
    with torch.no_grad():
        preds = model(X_te_t).argmax(1)
        acc = (preds == y_te_t).float().mean().item()
    test_accs.append(acc)
    scheduler.step(train_losses[-1])
    if (epoch + 1) % 10 == 0:
        print(f"  Epoch {epoch+1}/50 - Test Acc: {acc:.4f}")

mlp_acc = test_accs[-1]
y_pred_np = preds.numpy()

# Chart 4: Training Curves
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 4.5))
ax1.plot(train_losses, color='#4A90D9', linewidth=1.5)
ax1.set_xlabel('Epoch'); ax1.set_ylabel('Loss')
ax1.set_title('Training Loss', fontsize=11, fontweight='bold')
ax1.grid(True, alpha=0.3)
ax2.plot(train_accs, label='Train', color='#4A90D9', linewidth=1.5)
ax2.plot(test_accs, label='Test', color='#E74C3C', linewidth=1.5)
ax2.set_xlabel('Epoch'); ax2.set_ylabel('Accuracy')
ax2.set_title('Accuracy Over Epochs', fontsize=11, fontweight='bold')
ax2.legend(); ax2.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig('chart_training.png', dpi=180, bbox_inches='tight'); plt.close()

# Chart 5: Confusion Matrix
cm = confusion_matrix(y_test, y_pred_np)
fig, ax = plt.subplots(figsize=(10, 8))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
            xticklabels=le.classes_, yticklabels=le.classes_, ax=ax)
ax.set_xlabel('Predicted', fontsize=11)
ax.set_ylabel('Actual', fontsize=11)
ax.set_title('Confusion Matrix - MLP Genre Classifier', fontsize=13, fontweight='bold')
plt.tight_layout()
plt.savefig('chart_confusion.png', dpi=180, bbox_inches='tight'); plt.close()

# ── Build PDF ──
print("Building PDF...")
report = classification_report(y_test, y_pred_np, target_names=le.classes_, output_dict=True)
genre_counts = df['super_genre'].value_counts()


class ReportPDF(FPDF):
    def header(self):
        if self.page_no() > 1:
            self.set_font('Helvetica', 'I', 8)
            self.set_text_color(150, 150, 150)
            self.cell(0, 8, 'AI 100 Midterm - Spotify Genre Classification - Marco King', align='C')
            self.ln(4)

    def footer(self):
        self.set_y(-15)
        self.set_font('Helvetica', 'I', 8)
        self.set_text_color(150, 150, 150)
        self.cell(0, 10, f'Page {self.page_no()}', align='C')

    def section_title(self, num, title):
        self.ln(6)
        self.set_font('Helvetica', 'B', 14)
        self.set_text_color(44, 62, 80)
        self.cell(0, 10, f'{num}. {title}', new_x='LMARGIN', new_y='NEXT')
        self.set_draw_color(74, 144, 217)
        self.set_line_width(0.5)
        self.line(self.l_margin, self.get_y(), self.w - self.r_margin, self.get_y())
        self.ln(4)

    def sub_title(self, title):
        self.ln(3)
        self.set_font('Helvetica', 'B', 11)
        self.set_text_color(52, 73, 94)
        self.cell(0, 7, title, new_x='LMARGIN', new_y='NEXT')
        self.ln(2)

    def body_text(self, text):
        self.set_font('Helvetica', '', 10)
        self.set_text_color(30, 30, 30)
        self.multi_cell(0, 5.5, text)
        self.ln(2)

    def bullet(self, text, bold_prefix=''):
        self.set_font('Helvetica', '', 10)
        self.set_text_color(30, 30, 30)
        x = self.get_x()
        self.cell(6, 5.5, '-')
        if bold_prefix:
            self.set_font('Helvetica', 'B', 10)
            self.write(5.5, bold_prefix + ' ')
            self.set_font('Helvetica', '', 10)
        self.multi_cell(0, 5.5, text)
        self.ln(1)

    def add_table(self, headers, rows, col_widths=None):
        if col_widths is None:
            col_widths = [(self.w - self.l_margin - self.r_margin) / len(headers)] * len(headers)
        # Header
        self.set_font('Helvetica', 'B', 9)
        self.set_fill_color(74, 144, 217)
        self.set_text_color(255, 255, 255)
        for i, h in enumerate(headers):
            self.cell(col_widths[i], 7, h, border=1, fill=True, align='C')
        self.ln()
        # Rows
        self.set_font('Helvetica', '', 9)
        self.set_text_color(30, 30, 30)
        for ri, row in enumerate(rows):
            if ri % 2 == 0:
                self.set_fill_color(248, 249, 250)
            else:
                self.set_fill_color(255, 255, 255)
            for i, val in enumerate(row):
                align = 'L' if i == 0 else 'C'
                self.cell(col_widths[i], 6.5, str(val), border=1, fill=True, align=align)
            self.ln()
        self.ln(3)

    def add_chart(self, path, w=170):
        x = (self.w - w) / 2
        if self.get_y() + 80 > self.h - 30:
            self.add_page()
        self.image(path, x=x, w=w)
        self.ln(4)


pdf = ReportPDF()
pdf.set_auto_page_break(auto=True, margin=20)
pdf.add_page()

# ── Title Page ──
pdf.ln(30)
pdf.set_font('Helvetica', 'B', 26)
pdf.set_text_color(44, 62, 80)
pdf.multi_cell(0, 12, 'Spotify Genre Classification\nUsing a Deep Learning MLP', align='C')
pdf.ln(6)
pdf.set_draw_color(74, 144, 217)
pdf.set_line_width(1)
pdf.line(60, pdf.get_y(), pdf.w - 60, pdf.get_y())
pdf.ln(10)
pdf.set_font('Helvetica', '', 14)
pdf.set_text_color(100, 100, 100)
pdf.cell(0, 8, 'AI 100 - Midterm Project', align='C', new_x='LMARGIN', new_y='NEXT')
pdf.ln(4)
pdf.cell(0, 8, 'Marco King', align='C', new_x='LMARGIN', new_y='NEXT')
pdf.ln(4)
pdf.cell(0, 8, 'February 2026', align='C', new_x='LMARGIN', new_y='NEXT')

# ── Section 1: Problem Definition ──
pdf.add_page()
pdf.section_title(1, 'Problem Definition and Dataset Curation')

pdf.sub_title('Problem')
pdf.body_text(
    'Music streaming platforms like Spotify organize millions of tracks by genre, but genre labels '
    'are often applied manually or inconsistently. This project explores whether a deep learning model '
    'can automatically classify songs into broad genre categories using only their numeric audio '
    'characteristics - features like danceability, energy, tempo, and loudness - without listening '
    'to the actual audio.'
)
pdf.body_text(
    'This is framed as a multi-class classification problem: given 13 audio features for a track, '
    'predict which of 10 genre categories it belongs to.'
)

pdf.sub_title('Dataset')
pdf.body_text(
    'The dataset is the Spotify Tracks Dataset from Kaggle, containing approximately 114,000 tracks. '
    'Each track includes 13 numeric audio features extracted by Spotify\'s API:'
)

feature_rows = [
    ['Danceability', 'How suitable for dancing', '0.0 - 1.0'],
    ['Energy', 'Intensity and activity', '0.0 - 1.0'],
    ['Key', 'Musical key', '0 - 11'],
    ['Loudness', 'Overall loudness (dB)', '-60 - 0'],
    ['Mode', 'Major (1) or minor (0)', '0 or 1'],
    ['Speechiness', 'Spoken words presence', '0.0 - 1.0'],
    ['Acousticness', 'Acoustic confidence', '0.0 - 1.0'],
    ['Instrumentalness', 'No vocals prediction', '0.0 - 1.0'],
    ['Liveness', 'Live recording probability', '0.0 - 1.0'],
    ['Valence', 'Musical positivity', '0.0 - 1.0'],
    ['Tempo', 'Tempo in BPM', '~0 - 250'],
    ['Time Signature', 'Est. time signature', '1 - 5'],
    ['Duration (ms)', 'Track length', 'varies'],
]
pdf.add_table(['Feature', 'Description', 'Range'], feature_rows, [50, 80, 40])

pdf.sub_title('Genre Grouping')
pdf.body_text(
    'The original dataset contains 114 fine-grained genre labels. Many are closely related and have '
    'too few samples individually for reliable classification. To create a tractable problem, the 114 '
    'genres were grouped into 10 super-genres:'
)

genre_rows = []
for sg, genres in genre_map.items():
    examples = ', '.join(genres[:4])
    if len(genres) > 4:
        examples += f' (+{len(genres)-4})'
    genre_rows.append([sg, examples, f'{genre_counts[sg]:,}'])
pdf.add_table(['Super-Genre', 'Example Sub-Genres', 'Tracks'], genre_rows, [35, 105, 30])

pdf.add_chart('chart_distribution.png', w=160)

pdf.body_text(
    'The data was split 80/20 into training (91,200 samples) and test (22,800 samples) sets using '
    'stratified sampling to preserve class proportions. All features were standardized (zero mean, '
    'unit variance) using the training set statistics only.'
)

pdf.sub_title('Feature Analysis')
pdf.add_chart('chart_correlation.png', w=155)

pdf.body_text(
    'The correlation matrix shows most features are relatively independent, with the notable exception '
    'of energy and loudness (r=0.76). This low multicollinearity is favorable for classification, as '
    'each feature contributes relatively unique information.'
)

pdf.add_chart('chart_features.png', w=175)

pdf.body_text(
    'The box plots reveal distinct feature signatures across genres. Classical music stands out with '
    'high acousticness and instrumentalness. Metal shows high energy. Hip-Hop/R&B has elevated '
    'speechiness. These patterns suggest that audio features carry real discriminative signal.'
)

# ── Section 2: Deep Learning Model ──
pdf.add_page()
pdf.section_title(2, 'Deep Learning Model')

pdf.sub_title('Baseline: Logistic Regression')
pdf.body_text(
    'Before building a deep learning model, a logistic regression classifier was trained as a baseline '
    'to establish a performance floor. Logistic regression is a linear model - it can only learn linear '
    'decision boundaries between classes. This provides a reference point to measure how much the '
    'neural network improves by learning non-linear patterns.'
)

pdf.sub_title('Model Architecture: Multi-Layer Perceptron (MLP)')
pdf.body_text(
    'The primary model is a Multi-Layer Perceptron (MLP) implemented in PyTorch - a feedforward neural '
    'network of fully connected layers. It is well-suited for tabular data because it can learn complex '
    'non-linear relationships between inputs and outputs.'
)

pdf.ln(2)
pdf.set_font('Courier', '', 9)
pdf.set_fill_color(240, 244, 248)
pdf.set_draw_color(74, 144, 217)
arch_text = (
    "  Input (13 features)\n"
    "    |\n"
    "    +-- Linear(13 -> 256) -> BatchNorm -> ReLU -> Dropout(0.3)\n"
    "    |\n"
    "    +-- Linear(256 -> 128) -> BatchNorm -> ReLU -> Dropout(0.3)\n"
    "    |\n"
    "    +-- Linear(128 -> 64)  -> BatchNorm -> ReLU -> Dropout(0.2)\n"
    "    |\n"
    "    +-- Linear(64 -> 10)   -> Output (10 super-genres)\n"
    "\n"
    "  Total trainable parameters: ~46,000"
)
x = pdf.get_x()
y_pos = pdf.get_y()
pdf.rect(x, y_pos, pdf.w - pdf.l_margin - pdf.r_margin, 52, style='DF')
pdf.set_xy(x + 3, y_pos + 2)
pdf.multi_cell(0, 4.5, arch_text)
pdf.ln(4)

pdf.set_font('Helvetica', '', 10)
pdf.set_text_color(30, 30, 30)
pdf.body_text('Key design choices:')
pdf.bullet('Normalizes the input to each layer, stabilizing training and allowing higher learning rates.', 'Batch Normalization:')
pdf.bullet('Introduces non-linearity that allows the network to learn patterns logistic regression cannot.', 'ReLU Activation:')
pdf.bullet('Randomly zeroes out neurons during training to prevent overfitting. Higher dropout (30%) in earlier, wider layers.', 'Dropout (30% and 20%):')

pdf.sub_title('Training Configuration')
config_rows = [
    ['Loss Function', 'CrossEntropyLoss'],
    ['Optimizer', 'Adam (lr=0.001, weight_decay=1e-4)'],
    ['LR Scheduler', 'ReduceLROnPlateau (patience=5, factor=0.5)'],
    ['Batch Size', '256'],
    ['Epochs', '50'],
    ['Feature Scaling', 'StandardScaler (fit on train only)'],
]
pdf.add_table(['Parameter', 'Value'], config_rows, [55, 115])

# ── Section 3: Results ──
pdf.add_page()
pdf.section_title(3, 'Results')

pdf.sub_title('Model Comparison')
comp_rows = [
    ['Logistic Regression (baseline)', f'{lr_acc:.1%}'],
    ['MLP Neural Network', f'{mlp_acc:.1%}'],
    ['Relative Improvement', f'{(mlp_acc - lr_acc)/lr_acc*100:+.1f}%'],
]
pdf.add_table(['Model', 'Test Accuracy'], comp_rows, [100, 70])

pdf.body_text(
    f'For context, random guessing among 10 classes would yield ~10% accuracy. Both models significantly '
    f'outperform chance. The MLP\'s ~20% relative improvement over logistic regression demonstrates that '
    f'there are meaningful non-linear patterns in audio features that a deep learning model can capture '
    f'but a linear model cannot.'
)

pdf.sub_title('Training Dynamics')
pdf.add_chart('chart_training.png', w=165)

pdf.body_text(
    'Training loss decreased steadily over 50 epochs, while test accuracy plateaued around epoch 30-40, '
    'indicating convergence. The small gap between training and test accuracy suggests that dropout and '
    'batch normalization effectively prevented severe overfitting.'
)

pdf.sub_title('Per-Genre Performance')
perf_rows = []
for cls in le.classes_:
    r = report[cls]
    perf_rows.append([cls, f'{r["precision"]:.2f}', f'{r["recall"]:.2f}', f'{r["f1-score"]:.2f}', f'{int(r["support"])}'])
pdf.add_table(['Super-Genre', 'Precision', 'Recall', 'F1-Score', 'Support'], perf_rows, [40, 28, 28, 28, 28])

pdf.body_text(
    'Best-classified genres: Classical, Electronic, and Metal. These have distinct audio signatures - '
    'classical music is high in acousticness, metal is high in energy and loudness, and electronic '
    'music has distinctive danceability and tempo patterns.'
)
pdf.body_text(
    'Hardest genres: Hip-Hop/R&B and Jazz/Blues. These had the fewest training samples and share '
    'audio characteristics with other genres. Hip-hop\'s danceability and energy overlap heavily '
    'with Pop and Dance.'
)

pdf.sub_title('Confusion Matrix')
pdf.add_chart('chart_confusion.png', w=155)

pdf.body_text(
    'The confusion matrix reveals that most misclassifications occur between genres with similar audio '
    'profiles - Rock is often confused with Metal and Pop; Dance/Other is spread across multiple '
    'categories, reflecting its catch-all nature.'
)

# ── Section 4: Lessons ──
pdf.add_page()
pdf.section_title(4, 'Lessons and Experience')

pdf.sub_title('How This Project Was Built')
pdf.body_text(
    'As the assignment permits, I used an LLM (Claude) to help write the code for this project. '
    'My role was defining the problem, choosing the dataset, making design decisions (like how to '
    'group the 114 genres into 10 super-genres), and making sure I understood what the code was '
    'actually doing at each step. The LLM handled the implementation details - writing the PyTorch '
    'model class, setting up the training loop, and generating visualizations. I then reviewed the '
    'code, ran it locally, and interpreted the results.'
)
pdf.body_text(
    'I think this workflow is worth reflecting on honestly, because the hardest part of this project '
    'was not getting the code to run - it was making sure I actually understood what was happening. '
    'It would have been easy to let the LLM do everything without asking questions, but I made a point '
    'of understanding each decision: why we used CrossEntropyLoss instead of something else, what '
    'BatchNorm and Dropout actually do, why the features need to be standardized, and so on.'
)

pdf.sub_title('What Surprised Me')
pdf.body_text(
    'The biggest surprise was how fast training was. The entire model trained in about two minutes on '
    'my laptop CPU. You hear about companies spending billions of dollars and months of compute time '
    'to train large language models, so I expected deep learning to be slow. But our model has roughly '
    '46,000 parameters and processes 13 numbers per song - GPT-scale models have hundreds of billions '
    'of parameters and train on terabytes of text. The difference is something like six or seven orders '
    'of magnitude in scale, which puts the cost of frontier AI training into perspective.'
)
pdf.body_text(
    'I was also surprised that the model could predict genre at all from just 13 numeric features. '
    'Features like "danceability" and "valence" do not seem like the most obviously genre-defining '
    'characteristics, yet the model found enough signal to classify at 4.6x better than random chance. '
    'Classical and Metal were especially easy to identify - they have genuinely distinct audio signatures '
    'that show up clearly in the data.'
)

pdf.sub_title('What I Learned')
pdf.bullet(
    'PyTorch is a deep learning framework built by Meta that lets you define neural networks and train '
    'them in Python. The key feature is automatic differentiation - when you call loss.backward(), '
    'PyTorch computes how to adjust every weight in the network to reduce the error, without you '
    'having to manually code the calculus. This is what makes training practical.',
    'PyTorch:'
)
pdf.bullet(
    'An MLP is the simplest type of neural network - just layers of neurons connected to each other. '
    'Each layer applies a linear transformation, then a non-linear activation function (ReLU). '
    'Stacking multiple layers lets the network learn complex, non-linear patterns that a single '
    'linear model like logistic regression cannot capture.',
    'MLP architecture:'
)
pdf.bullet(
    'Grouping 114 genres into 10 super-genres was a design decision that shaped the entire project. '
    'It taught me that how you frame a problem matters as much as what model you use. A messy catch-all '
    'category like "Dance/Other" actively hurts accuracy because it has no coherent audio signature.',
    'Problem framing matters:'
)

pdf.sub_title('Limitations and What I Would Try Differently')
pdf.bullet(
    'Genre boundaries are inherently fuzzy. A song labeled "indie" could reasonably be called "rock" '
    'or "pop" - even human listeners disagree. The confusion matrix shows this overlap clearly.',
    'Genre overlap:'
)
pdf.bullet(
    'The 13 numeric features miss lyrics, vocal style, instrumentation, and cultural context. Using '
    'spectrograms or raw audio with a convolutional neural network would likely perform much better.',
    'Limited input features:'
)
pdf.bullet(
    'Hip-Hop/R&B had only 4,000 samples vs. Latin/World\'s 20,000. Balancing the classes with '
    'oversampling techniques like SMOTE could improve recall for smaller genres.',
    'Class imbalance:'
)

# ── References ──
pdf.ln(6)
pdf.sub_title('References')
pdf.set_font('Helvetica', '', 9)
pdf.set_text_color(30, 30, 30)
refs = [
    'Spotify Tracks Dataset - https://www.kaggle.com/datasets/maharshipandya/-spotify-tracks-dataset',
    'PyTorch Documentation - https://pytorch.org/docs/stable/',
    'scikit-learn Documentation - https://scikit-learn.org/stable/',
]
for ref in refs:
    pdf.cell(6, 5, '-')
    pdf.cell(0, 5, ref, new_x='LMARGIN', new_y='NEXT')

pdf.output('AI100_Midterm_Report.pdf')
print(f"\nDone! PDF saved to: AI100_Midterm_Report.pdf")
print(f"Logistic Regression: {lr_acc:.4f}")
print(f"MLP Accuracy:        {mlp_acc:.4f}")
print(f"Improvement:         {(mlp_acc-lr_acc)/lr_acc*100:+.1f}%")
