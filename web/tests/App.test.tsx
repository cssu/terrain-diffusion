// simple smoke test for web dev

import { render, screen } from '@testing-library/react'
import { expect, it } from 'vitest'
import App from '../src/App'

it('renders the page heading', () => {
  render(<App />)

  expect(screen.getByRole('heading', { name: 'terrain-diffusion' })).toBeInTheDocument()
})
