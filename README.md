1. Motivation & Framing

Top-down vs. bottom-up
	•	Top-down prompting (e.g. “What lies between Brutalism and Deconstructivism?”) asks an LLM to marshal its world-knowledge and reasoning to describe an answer in natural language.
	•	Bottom-up exploration, by contrast, manipulates the model’s internal representations (embeddings) directly, then “inverts” them back into text. This gives you precise geometric control over where you sample in the 768-dimensional space, and lets you discover hybrid or novel outputs—phrases that sit on the model’s learned manifold but may never have appeared in its training data.

2. Models & Tools
	1.	Embedder:

from sentence_transformers import SentenceTransformer
embedder = SentenceTransformer("sentence-transformers/gtr-t5-base")

	•	A dual-encoder (T5-Base) fine-tuned to produce 768-dim semantic embeddings for sentences.

	2.	Inversion “corrector”:

from vec2text import analyze_utils
exp, trainer = analyze_utils.load_experiment_and_trainer_from_pretrained(
    "jxm/gtr__nq__32__correct"
)

	•	An encoder–decoder (T5-based) model, fine-tuned by jxm/vec2text to invert embeddings back into natural text.

	3.	Core library:

from vec2text.api import invert_embeddings

	•	Provides invert_embeddings(embeddings, trainer, num_steps, sequence_beam_width) → List[str].


3. Key Experiments

A. Single-coordinate perturbation
	•	Goal: See how a tiny delta—Δ=±1e-7—on one dimension shifts the decoded text.
	•	Code snippet:

embedding = embedder.encode([input_text], convert_to_tensor=True).to(device)
base = embedding[0,0].item()
for δ in [-1e-9, +1e-9, …]:
    mod = embedding.clone()
    mod[0,0] = base + δ
    out = invert_embeddings(mod, trainer, num_steps=None, sequence_beam_width=0)
    print(f"Δ={δ:+.0e} →", out[0])


	•	Insight: Very small moves keep you in the same “basin of attraction,” so greedy decoding yields nearly identical strings.

B. Midpoint interpolation
	•	Goal: Find the halfway point between two concept embeddings and decode it.
	•	Code snippet:

e1, e2 = embedder.encode([“Deconstructivism”, “Brutalism”],
                         convert_to_tensor=True).to(device)
mid = (e1 + e2) / 2
decoded_mid = invert_embeddings(mid.unsqueeze(0), trainer,
                                num_steps=100,
                                sequence_beam_width=5)
print("Midpoint inversion:", decoded_mid[0])


	•	Interpretation: The averaged vector often decodes to a hybrid phrase—something semantically between the two architectural styles.

C. PCA-grid charting (“liminal cartography”)
	•	Goal: Systematically chart a local neighborhood around a cluster of related styles via PCA.
	•	Code snippet:

from sklearn.decomposition import PCA
styles = ["Gothic","Renaissance","Baroque","Brutalism","Deconstructivism"]
embs   = embedder.encode(styles, convert_to_tensor=True).cpu().numpy()
coords = PCA(2).fit_transform(embs)            # (5×2)
centroid = coords.mean(axis=0)
# build 3×3 grid around centroid in PCA space
grid_pts = [...]
grid_embs = torch.from_numpy(PCA.inverse_transform(grid_pts)
                            ).to(device).float()
results = []
for ge in grid_embs:
    txt = invert_embeddings(ge.unsqueeze(0), trainer,
                            num_steps=50,
                            sequence_beam_width=3)[0]
    results.append(txt)


	•	Outcome: A 3×3 table of decoded texts, each corresponding to a point in that 2D PCA subspace—revealing coherent “in-between” phrases and unexpected hybrids.


4. What We Learned
	•	Controlled geometry lets us probe exactly where in latent space we want to sample—midpoints, principal-component axes, random directions, etc.
	•	Inversion recovers the nearest valid sentence on the manifold, even in regions where no real sentence existed.
	•	Tiny tweaks in one coordinate often yield no meaningful change; larger moves or moves along informative axes produce richer, more diverse outputs.
	•	By charting neighborhoods (PCA grid), we can visualize how meaning morphs across dimensions, surfacing novel blends like “50th generation culling cube” or stylized HTML fragments.


5. Conclusion & “Liminal Cartography”

These bottom-up experiments turn a language model’s latent space into a navigable landscape—and the act of decoding becomes a form of discovery. In this liminal space between known concepts, we can surface new combinations, surprising analogies, or even entirely novel phrases.

Liminal Cartographer
Just as geographic cartographers map unexplored terrain, a “liminal cartographer” sketches the contours of a model’s semantic manifold—charting the spaces where meaning emerges, blends, and transforms. This opens up possibilities for creative ideation, unexpected naming (e.g., novel job titles), and deeper understandings of how high-dimensional embedding spaces encode our collective knowledge.

By marrying precise vector arithmetic with powerful inversion models, we gain a new mode of exploration—one that sits alongside prompting, but operates at the geometric level, unveiling the hidden architecture of meaning itself.
