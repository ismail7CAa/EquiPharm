# ---
# jupyter:
#   jupytext:
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#       jupytext_version: 1.19.1
#   kernelspec:
#     display_name: Python (dig_envi)
#     language: python
#     name: dig_envi
# ---

# %%
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, Circle, Polygon
import numpy as np

def draw_hex_molecule(ax, center, radius=0.32, linecolor="#1f2937", lw=2.2):
    cx, cy = center
    angles = np.linspace(0, 2 * np.pi, 6, endpoint=False) + np.pi / 6
    pts = np.c_[cx + radius * np.cos(angles), cy + radius * np.sin(angles)]

    for i in range(6):
        x1, y1 = pts[i]
        x2, y2 = pts[(i + 1) % 6]
        ax.plot([x1, x2], [y1, y2], color=linecolor, lw=lw, solid_capstyle="round", zorder=2)

    for x, y in pts:
        atom = Circle((x, y), radius * 0.11, facecolor="white", edgecolor=linecolor, lw=1.8, zorder=3)
        ax.add_patch(atom)

    aromatic = Circle((cx, cy), radius * 0.42, facecolor="none", edgecolor=linecolor, lw=1.4, zorder=2)
    ax.add_patch(aromatic)


def draw_rdkit_icon(ax, center, scale=1.0, color="#0f766e"):
    cx, cy = center
    pts = np.array([
        [cx - 0.23 * scale, cy + 0.12 * scale],
        [cx + 0.00 * scale, cy + 0.25 * scale],
        [cx + 0.25 * scale, cy + 0.05 * scale],
        [cx + 0.10 * scale, cy - 0.20 * scale],
        [cx - 0.18 * scale, cy - 0.15 * scale],
    ])

    edges = [(0, 1), (1, 2), (2, 3), (3, 4), (4, 0), (1, 3)]
    for i, j in edges:
        ax.plot([pts[i, 0], pts[j, 0]], [pts[i, 1], pts[j, 1]], color=color, lw=2.0, solid_capstyle="round", zorder=2)

    for x, y in pts:
        ax.add_patch(Circle((x, y), 0.045 * scale, facecolor="white", edgecolor=color, lw=1.8, zorder=3))


def draw_graph_icon(ax, center, scale=1.0, color="#2563eb"):
    cx, cy = center
    pts = np.array([
        [cx - 0.22 * scale, cy + 0.12 * scale],
        [cx + 0.22 * scale, cy + 0.15 * scale],
        [cx - 0.05 * scale, cy - 0.02 * scale],
        [cx + 0.18 * scale, cy - 0.20 * scale],
        [cx - 0.25 * scale, cy - 0.22 * scale],
    ])

    edges = [(0, 1), (0, 2), (2, 1), (2, 3), (2, 4), (4, 3)]
    for i, j in edges:
        ax.plot([pts[i, 0], pts[j, 0]], [pts[i, 1], pts[j, 1]], color=color, lw=2.0, solid_capstyle="round", zorder=2)

    for x, y in pts:
        ax.add_patch(Circle((x, y), 0.05 * scale, facecolor="white", edgecolor=color, lw=1.8, zorder=3))


def draw_encoder_icon(ax, center, scale=1.0, color="#7c3aed"):
    cx, cy = center
    widths = [0.50, 0.40, 0.30]
    ys = [0.16, 0.0, -0.16]
    for w, yoff in zip(widths, ys):
        box = FancyBboxPatch(
            (cx - w / 2, cy + yoff - 0.05),
            w,
            0.10,
            boxstyle="round,pad=0.01,rounding_size=0.02",
            facecolor="white",
            edgecolor=color,
            lw=1.9,
            zorder=2,
        )
        ax.add_patch(box)


def draw_embedding_icon(ax, center, scale=1.0, color="#db2777"):
    cx, cy = center
    xs = np.linspace(cx - 0.22 * scale, cx + 0.22 * scale, 6)
    hs = [0.10, 0.22, 0.14, 0.28, 0.18, 0.24]
    for x, h in zip(xs, hs):
        ax.plot([x, x], [cy - 0.16 * scale, cy - 0.16 * scale + h * scale], color=color, lw=3.0, zorder=2)
        ax.add_patch(Circle((x, cy - 0.16 * scale + h * scale), 0.018 * scale, facecolor=color, edgecolor=color, zorder=3))


def draw_similarity_icon(ax, center, scale=1.0, color="#ea580c"):
    cx, cy = center
    ax.plot([cx, cx + 0.28 * scale], [cy, cy], color=color, lw=2.4, zorder=2)
    ax.plot([cx, cx + 0.22 * scale], [cy, cy + 0.18 * scale], color=color, lw=2.4, zorder=2)
    ax.add_patch(Circle((cx, cy), 0.02 * scale, facecolor=color, edgecolor=color, zorder=3))
    theta = np.linspace(0, np.deg2rad(40), 30)
    r = 0.10 * scale
    ax.plot(cx + r * np.cos(theta), cy + r * np.sin(theta), color=color, lw=1.6, zorder=2)


def draw_blackbox_icon(ax, center, scale=1.0, color="#4f46e5"):
    cx, cy = center
    box = FancyBboxPatch(
        (cx - 0.22 * scale, cy - 0.18 * scale),
        0.44 * scale,
        0.36 * scale,
        boxstyle="round,pad=0.02,rounding_size=0.04",
        facecolor="white",
        edgecolor=color,
        lw=1.8,
        zorder=2,
    )
    ax.add_patch(box)
    ax.text(cx, cy, "f(x)", ha="center", va="center", fontsize=10, color=color, fontweight="bold", zorder=3)


def draw_torsion_icon(ax, center, scale=1.0, color="#0891b2"):
    cx, cy = center
    pts = np.array([
        [cx - 0.22 * scale, cy],
        [cx - 0.05 * scale, cy + 0.12 * scale],
        [cx + 0.08 * scale, cy - 0.10 * scale],
        [cx + 0.24 * scale, cy + 0.05 * scale],
    ])

    for i in range(3):
        ax.plot([pts[i, 0], pts[i + 1, 0]], [pts[i, 1], pts[i + 1, 1]], color=color, lw=2.0, zorder=2)

    for x, y in pts:
        ax.add_patch(Circle((x, y), 0.035 * scale, facecolor="white", edgecolor=color, lw=1.5, zorder=3))

    theta = np.linspace(np.pi / 6, np.pi * 0.95, 40)
    r = 0.12 * scale
    ax.plot(cx + r * np.cos(theta), cy + 0.18 * scale + r * np.sin(theta), color=color, lw=1.4, zorder=2)


def add_shadowed_box(ax, x, y, w, h, title, subtitle, edgecolor, icon_func, icon_color, facecolor="white"):
    shadow = FancyBboxPatch(
        (x + 0.06, y - 0.06),
        w,
        h,
        boxstyle="round,pad=0.03,rounding_size=0.18",
        facecolor="#cbd5e1",
        edgecolor="none",
        alpha=0.25,
        zorder=0,
    )
    ax.add_patch(shadow)

    card = FancyBboxPatch(
        (x, y),
        w,
        h,
        boxstyle="round,pad=0.03,rounding_size=0.18",
        facecolor=facecolor,
        edgecolor=edgecolor,
        lw=2.0,
        zorder=1,
    )
    ax.add_patch(card)

    icon_center = (x + 0.55, y + h / 2)
    icon_func(ax, icon_center, scale=1.05, color=icon_color)

    ax.text(
        x + 1.05,
        y + h * 0.63,
        title,
        fontsize=12,
        fontweight="bold",
        ha="left",
        va="center",
        color="#111827",
        zorder=4,
    )
    ax.text(
        x + 1.05,
        y + h * 0.34,
        subtitle,
        fontsize=9.6,
        ha="left",
        va="center",
        color="#4b5563",
        zorder=4,
    )


def add_embedding_node(ax, center, label, edgecolor="#be185d", icon_color="#db2777"):
    cx, cy = center
    w, h = 2.9, 1.2

    shadow = FancyBboxPatch(
        (cx - w / 2 + 0.06, cy - h / 2 - 0.06),
        w,
        h,
        boxstyle="round,pad=0.03,rounding_size=0.20",
        facecolor="#cbd5e1",
        edgecolor="none",
        alpha=0.25,
        zorder=0,
    )
    ax.add_patch(shadow)

    box = FancyBboxPatch(
        (cx - w / 2, cy - h / 2),
        w,
        h,
        boxstyle="round,pad=0.03,rounding_size=0.20",
        facecolor="white",
        edgecolor=edgecolor,
        lw=2.0,
        zorder=1,
    )
    ax.add_patch(box)

    draw_embedding_icon(ax, (cx - 0.78, cy), scale=1.1, color=icon_color)
    ax.text(cx - 0.25, cy + 0.14, label, fontsize=11.4, fontweight="bold", ha="left", va="center", color="#111827")
    ax.text(cx - 0.25, cy - 0.16, "latent vector", fontsize=9.5, ha="left", va="center", color="#4b5563")


def add_similarity_diamond(ax, center, label="Cosine similarity"):
    cx, cy = center
    w, h = 3.2, 1.9

    shadow_pts = np.array([
        [cx, cy + h / 2 - 0.06],
        [cx + w / 2 + 0.06, cy - 0.06],
        [cx, cy - h / 2 - 0.06],
        [cx - w / 2 + 0.06, cy - 0.06],
    ])
    ax.add_patch(Polygon(shadow_pts, closed=True, facecolor="#cbd5e1", edgecolor="none", alpha=0.25, zorder=0))

    pts = np.array([
        [cx, cy + h / 2],
        [cx + w / 2, cy],
        [cx, cy - h / 2],
        [cx - w / 2, cy],
    ])
    ax.add_patch(Polygon(pts, closed=True, facecolor="white", edgecolor="#c2410c", lw=2.0, zorder=1))

    # Icon centered ABOVE
    draw_similarity_icon(ax, (cx, cy + 0.40), scale=1.0, color="#ea580c")
    
    # Title centered
    ax.text(
        cx,
        cy + 0.05,
        label,
        fontsize=11.6,
        fontweight="bold",
        ha="center",
        va="center",
        color="#111827"
    )
    
    # Subtitle centered
    ax.text(
        cx,
        cy - 0.35,
        "embedding comparison",
        fontsize=9.2,
        ha="center",
        va="center",
        color="#4b5563"
    )


def add_output_box(ax, x, y, w, h):
    shadow = FancyBboxPatch(
        (x + 0.06, y - 0.06),
        w,
        h,
        boxstyle="round,pad=0.03,rounding_size=0.18",
        facecolor="#cbd5e1",
        edgecolor="none",
        alpha=0.25,
        zorder=0,
    )
    ax.add_patch(shadow)

    box = FancyBboxPatch(
        (x, y),
        w,
        h,
        boxstyle="round,pad=0.03,rounding_size=0.18",
        facecolor="white",
        edgecolor="#0369a1",
        lw=2.0,
        zorder=1,
    )
    ax.add_patch(box)

    ax.text(x + w / 2, y + h * 0.62, "Optimal similarity", fontsize=12, fontweight="bold", ha="center", va="center", color="#111827")
    ax.text(x + w / 2, y + h * 0.33, "score / ranking", fontsize=9.5, ha="center", va="center", color="#4b5563")


def draw_orthogonal_arrow(ax, points, color="#64748b", lw=2.2, head_len=0.16, head_width=0.11):
    """
    Draw a flowchart-style orthogonal connector.
    points: list of (x, y) tuples
    Arrow head is placed on the final segment end.
    """
    xs = [p[0] for p in points]
    ys = [p[1] for p in points]

    # Draw all segments except the very last one fully
    for i in range(len(points) - 2):
        ax.plot([xs[i], xs[i+1]], [ys[i], ys[i+1]], color=color, lw=lw, solid_capstyle="butt", zorder=1.5)

    # Final segment shortened for arrow head
    x0, y0 = points[-2]
    x1, y1 = points[-1]

    dx = x1 - x0
    dy = y1 - y0
    length = np.hypot(dx, dy)
    if length == 0:
        return

    ux, uy = dx / length, dy / length
    xb = x1 - ux * head_len
    yb = y1 - uy * head_len

    ax.plot([x0, xb], [y0, yb], color=color, lw=lw, solid_capstyle="butt", zorder=1.5)

    # Arrow head triangle
    px, py = -uy, ux
    tip = np.array([x1, y1])
    left = np.array([xb, yb]) + np.array([px, py]) * head_width
    right = np.array([xb, yb]) - np.array([px, py]) * head_width

    head = Polygon([tip, left, right], closed=True, facecolor=color, edgecolor=color, zorder=1.6)
    ax.add_patch(head)


# Build figure
fig, ax = plt.subplots(figsize=(27, 11), dpi=500)
ax.set_xlim(0, 24)
ax.set_ylim(0, 12)
ax.axis("off")

fig.patch.set_facecolor("white")
ax.set_facecolor("#f8fafc")

ax.text(
    12,
    11.2,
    "Pharmacophore Screening Pipeline ",
    fontsize=21,
    fontweight="bold",
    ha="center",
    va="center",
    color="#0f172a",
)


ax.text(1.2, 8.9, "Query branch", fontsize=12, fontweight="bold", color="#334155")
ax.text(1.2, 3.7, "Candidate branch", fontsize=12, fontweight="bold", color="#334155")

# Molecule icons
draw_hex_molecule(ax, (1.5, 7.75), radius=0.5, linecolor="#0f172a")
ax.text(1.5, 6.88, "Query\nmolecule", fontsize=11.2, ha="center", va="center", color="#111827", fontweight="bold")

draw_hex_molecule(ax, (1.5, 2.55), radius=0.5, linecolor="#0f172a")
ax.text(1.5, 1.68, "Candidate\nmolecule", fontsize=11.2, ha="center", va="center", color="#111827", fontweight="bold")

# Larger box geometry
w, h = 3.2, 1.45
x1, x2, x3 = 2.8, 6.9, 11.0
qy = 6.95
cy = 1.75

# Query path
add_shadowed_box(ax, x1, qy, w, h, "RDKit molecule", "sanitization / features", "#0f766e", draw_rdkit_icon, "#0f766e")
add_shadowed_box(ax, x2, qy, w, h, "PyG graph", "nodes / edges / attributes", "#2563eb", draw_graph_icon, "#2563eb")
add_shadowed_box(ax, x3, qy, w, h, "3D-Encoder", "geometric representation model", "#7c3aed", draw_encoder_icon, "#7c3aed")

# Candidate path
add_shadowed_box(ax, x1, cy, w, h, "RDKit molecule", "updated candidate conformer", "#0f766e", draw_rdkit_icon, "#0f766e")
add_shadowed_box(ax, x2, cy, w, h, "PyG graph", "nodes / edges / attributes", "#2563eb", draw_graph_icon, "#2563eb")
add_shadowed_box(ax, x3, cy, w, h, "3D-Encoder", "geometric representation model", "#7c3aed", draw_encoder_icon, "#7c3aed")

# Embeddings
add_embedding_node(ax, (16.2, 7.75), "Query embedding")
add_embedding_node(ax, (16.2, 2.55), "Candidate embedding")

# Similarity and output
add_similarity_diamond(ax, (20.0, 5.15), "Cosine similarity")
add_output_box(ax, 22.0, 4.45, 1.9, 1.4)

# Forward arrows, query
draw_orthogonal_arrow(ax, [(1.95, 7.75), (2.8, 7.75)])
draw_orthogonal_arrow(ax, [(6.0, 7.75), (6.9, 7.75)])
draw_orthogonal_arrow(ax, [(10.1, 7.75), (11.0, 7.75)])
draw_orthogonal_arrow(ax, [(14.2, 7.75), (14.75, 7.75)])

# Forward arrows, candidate
draw_orthogonal_arrow(ax, [(1.95, 2.55), (2.8, 2.55)])
draw_orthogonal_arrow(ax, [(6.0, 2.55), (6.9, 2.55)])
draw_orthogonal_arrow(ax, [(10.1, 2.55), (11.0, 2.55)])
draw_orthogonal_arrow(ax, [(14.2, 2.55), (14.75, 2.55)])

# Embeddings to cosine similarity: orthogonal and exact
draw_orthogonal_arrow(ax, [(17.65, 7.75), (18.6, 7.75), (18.6, 5.85), (19.0, 5.85)])
draw_orthogonal_arrow(ax, [(17.65, 2.55), (18.6, 2.55), (18.6, 4.45), (19.0, 4.45)])

# Cosine similarity to output
draw_orthogonal_arrow(ax, [(21.6, 5.15), (22.0, 5.15)])

# Optimization loop blocks
loop_w, loop_h = 3.55, 1.25
opt_x1, opt_x2 = 14.2, 6.9
opt_y = 0.05

add_shadowed_box(
    ax,
    opt_x1,
    opt_y,
    loop_w,
    loop_h,
    "Black-box optimization",
    "maximize cosine similarity",
    "#4f46e5",
    draw_blackbox_icon,
    "#4f46e5",
)

add_shadowed_box(
    ax,
    opt_x2,
    opt_y,
    loop_w,
    loop_h,
    "Angle / torsion optimization",
    "conformer refinement",
    "#0891b2",
    draw_torsion_icon,
    "#0891b2",
)

# Orthogonal optimization loop
# Cosine similarity -> black-box optimization
draw_orthogonal_arrow(ax, [
    (20.0, 4.2),    # start at cosine similarity
    (20.0, 0.755),  # go straight DOWN
    (17.8, 0.725)   # then LEFT into black-box optimization
])

# Black-box optimization -> torsion optimization
draw_orthogonal_arrow(ax, [
    (14.15, 0.725),
    (10.5, 0.725)
])

# Torsion optimization -> RDKit candidate molecule
draw_orthogonal_arrow(ax, [
    (6.88, 0.725),   # from torsion box (right side)
    (6.8, 0.725),   # go left
    (4.0, 0.75),    # go up to RDKit arrow level
    (4.0, 1.7)     # STOP on the RDKit arrow line (not the box)
])



plt.tight_layout()
plt.savefig("pipeline_with_optimization_loop_orthogonal.png", dpi=500, bbox_inches="tight")
plt.savefig("pipeline_with_optimization_loop_orthogonal.svg", bbox_inches="tight")
plt.show()

# %%
