export type Colour = [number, number, number]

export const MIN_HEIGHT = 0
export const MAX_HEIGHT = 255

const STOPS: { height: number; colour: Colour }[] = [
  { height: 0, colour: [12, 44, 92] },
  { height: 80, colour: [60, 120, 180] },
  { height: 100, colour: [214, 198, 138] },
  { height: 150, colour: [74, 128, 64] },
  { height: 200, colour: [120, 104, 88] },
  { height: 255, colour: [248, 248, 252] },
]

function mix(from: Colour, to: Colour, along: number): Colour {
  return [0, 1, 2].map((channel) =>
    Math.round(from[channel] + (to[channel] - from[channel]) * along),
  ) as Colour
}

export function heightColour(height: number): Colour {
  const clamped = Math.min(Math.max(height, MIN_HEIGHT), MAX_HEIGHT)

  for (let i = 1; i < STOPS.length; i++) {
    const start = STOPS[i - 1]
    const end = STOPS[i]
    if (clamped <= end.height) {
      return mix(start.colour, end.colour, (clamped - start.height) / (end.height - start.height))
    }
  }

  return STOPS[STOPS.length - 1].colour
}

export function gridColours(heights: ArrayLike<number>): Uint8Array {
  const colours = new Uint8Array(heights.length * 3)

  for (let i = 0; i < heights.length; i++) {
    colours.set(heightColour(heights[i]), i * 3)
  }

  return colours
}
