/** Generate a string ID like "Environment1", "Agent3", etc. */
export function nextStringId(prefix: string, existingIds: string[]): string {
  const nums = existingIds
    .filter(id => id.startsWith(prefix))
    .map(id => parseInt(id.slice(prefix.length), 10))
    .filter(n => !isNaN(n))
  const next = nums.length > 0 ? Math.max(...nums) + 1 : 1
  return `${prefix}${next}`
}
