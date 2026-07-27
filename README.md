# terrain-diffusion

A reproduction of the InfiniteDiffusion paper: generating realistic landscape terrain that continues forever in every direction, comes out the same every time from the same seed, and uses a fixed amount of memory no matter how far you explore.


## Getting set up

Install [uv](https://docs.astral.sh/uv/getting-started/installation/), the tool used to manage Python and Python packages:

```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
```

Then install the project's packages:

```bash
scripts/install-dependencies.sh
```

> You do not need to install python, uv handles and maintains the correct python version.

## Contributing

See [CONTRIBUTING.md](CONTRIBUTING.md) for how to run the checks, how tests are organised, and how changes get reviewed and merged.

## TODO

- Add end to end tests with [Playwright](https://playwright.dev/) once the visualizer draws something. The current tests run in jsdom, which has no canvas and no WebGL, thus cannot check what appears on screen
- Playwright simulates a real browser and can compare screenshots