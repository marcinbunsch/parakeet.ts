/**
 * The backend boundary.
 *
 * The library is abstracted at the *model* boundary, not the tensor boundary:
 * everything except the encoder forward and the prediction+joint step is plain
 * TypeScript and shared by all backends.
 *
 * Two calls are enough. The ONNX export fuses the prediction network and the
 * joint network into a single `decoder_joint` graph that cannot be split, so
 * `decodeStep` is the unit of work — the MLX backend simply calls its own
 * `predict` and `joint` back to back behind it.
 */
export {};
//# sourceMappingURL=backend.js.map