/** Decode compact little-endian trajectories produced by paperleet.common.viz. */
export function decodeViz(value: unknown): any {
  if (Array.isArray(value)) return value.map(decodeViz);
  if (!value || typeof value !== "object") return value;
  const object = value as Record<string, any>;
  if (object.encoding === "i16-base64") {
    const binary = atob(object.data);
    const bytes = Uint8Array.from(binary, (c) => c.charCodeAt(0));
    const view = new DataView(bytes.buffer);
    let offset = 0;
    const unpack = (dimensions: number[]): any[] =>
      Array.from({ length: dimensions[0] }, () => {
        if (dimensions.length > 1) return unpack(dimensions.slice(1));
        const number = view.getInt16(offset, true) / object.scale;
        offset += 2;
        return number;
      });
    return unpack(object.shape);
  }
  return Object.fromEntries(
    Object.entries(object).map(([key, child]) => [key, decodeViz(child)]),
  );
}
