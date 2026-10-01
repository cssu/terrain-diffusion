// simple smoke test for web dev

import { render } from '@testing-library/react'
import { expect, it } from 'vitest'
import App from '../src/App'

it('draws the terrain view', () => {
  const { container } = render(<App />)

  expect(container.querySelector('canvas')).toBeInTheDocument()
})
