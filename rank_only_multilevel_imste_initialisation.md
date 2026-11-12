# Rank-only multilevel initialization for IMSTE

## Aim

Use the existing ranked IMSTE graph to build a useful initial layout quickly. Coarsening uses edge ranks only; it does not use the original input distances. This is a design proposal, not an implemented or evaluated feature.

## Graph and edge scores

Let the current graph be (G_0=(V_0,E_0)). For each edge (e=(i,j)) with rank (r_e), define its coarsening score as

\[
a_e = \frac{1}{r_e}.
\]

These are the inverse-rank weights used for IMSTE's positive-edge attraction. Input distances are not used during coarsening. See the [IMSTE white paper](WHITEPAPER.md) for the rank-weighted objective.

## One coarsening level

Given a graph with (N) vertices:

1. **Set a target number of groups.** Start with (K=\lfloor N/4\rfloor), adjusted so that each connected component has at least one group and (K\geq K_{\min}). The reduction ratio and minimum size are tunable. (K) is a target number of supernodes; groups do not need to contain exactly four vertices.

2. **Choose seed vertices.** Score each vertex by its weighted degree:
   
   \[
   h(i)=\sum_{e\ni i} a_e.
   \]
   
   Select about (K) high-scoring seeds, with at least one seed per connected component. When scores are similar, prefer seeds that are not all immediate neighbours, so one local region does not receive all the seeds.

3. **Propagate group labels.** Give each seed a distinct group label. In rounds, each unlabelled vertex (i) totals the scores of its edges to already labelled neighbours in each group (C):
   
   \[
   v(i,C)=\sum_{e=(i,j):\,j\in C} a_e.
   \]
   
   Assign (i) to the group with the largest vote. Resolve ties deterministically, using a fixed vertex order or the random seed. Continue until all reachable vertices have labels or a round makes no assignments. Keep disconnected components separate; give any isolated or otherwise unassigned vertex its own group.

4. **Build the quotient graph.** Each group becomes a supernode. For every pair of distinct groups (A,B), sum the scores of original edges crossing between them:
   
   \[
   W_{AB}=\sum_{e=(i,j):\,i\in A,\,j\in B} a_e.
   \]
   
   Drop internal edges from the coarse layout graph because their endpoints now share a coordinate. Keep the vertex-to-group map and each group's size. A rank histogram can be retained for diagnostics; (W_{AB}) is an aggregate attraction weight, not a newly computed MST rank.

## Recursive layout and expansion

Repeat coarsening until the graph has at most a chosen base size. Initialize and lay out that smallest graph using its aggregated attraction weights, then expand one level at a time.

For each fine-level vertex (i) in group (C), let (y_C) be its group's coarse coordinate. Estimate a local position from the coordinates of neighbouring groups:

\[
\bar y_i =
\frac{\sum_{e=(i,j),\,g(j)\ne C} a_e\,y_{g(j)}}
     {\sum_{e=(i,j),\,g(j)\ne C} a_e}.
\]

If the denominator is nonzero, initialize (y_i=\tfrac12(y_C+\bar y_i)). Otherwise, place it near (y_C) with a small seeded offset. Run a short refinement at this level, then expand to the next finer level. At the finest level, refine with the original vertices, edges, ranks, and existing IMSTE objective.

## Pseudocode

```text
levels = [original_ranked_graph]
while number_of_vertices(levels[-1]) > base_size:
    G = levels[-1]
    K = target_group_count(G, reduction_ratio, base_size)
    seeds = choose_high_weighted_degree_seeds(G, K, one_per_component=True)
    labels = propagate_labels(G, seeds, edge_score=1/rank)
    levels.append(aggregate_cross_group_edges(G, labels, score=sum(1/rank)))

Y = initialise_and_layout(levels[-1])
for each finer level G in reverse(levels):
    Y = expand_from_group_and_neighbour_coordinates(G, Y, edge_score=1/rank)
    Y = short_refinement(G, Y)
Y = full_IMSTE_refinement(original_ranked_graph, Y)
```

## Initial experiment

Start with a reduction ratio of four and a conservative base size. Compare the resulting initialization with the current PCA and random starts, using the same original IMSTE graph, optimizer settings, final epoch budget, and matched random seeds where applicable. Record initialization cost, objective before and after refinement, epochs to reach a target objective, neighbourhood preservation, and a global-structure measure.

This rank-only design gives every rank-one edge the same coarsening score, including long bridges. Measure whether that leads to undesirable merges before treating the method as useful.

## Relation to UMAP

UMAP's recursive initialization uses hub-seeded label propagation and recursively expands and refines its embedding. This proposal adapts that coarsening pattern to IMSTE's inverse-rank edge scores; it does not use UMAP's fuzzy graph weights. See the [UMAP recursive initialization documentation](https://umap-learn.readthedocs.io/en/latest/api.html#recursive-initialization).
