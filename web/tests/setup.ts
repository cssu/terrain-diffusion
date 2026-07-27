// Runs before every test file and clears the rendered page 
// between tests so one test cannot affect the next

import '@testing-library/jest-dom/vitest'
import { cleanup } from '@testing-library/react'
import { afterEach } from 'vitest'

afterEach(() => {
  cleanup()
})
