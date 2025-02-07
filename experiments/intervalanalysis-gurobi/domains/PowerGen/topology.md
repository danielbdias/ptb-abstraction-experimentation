```mermaid
graph TD
    temperature
    p1
    p2
    p3
    p4
    p5

    subgraph powerplants
        p1
        p2
        p3
        p4
        p5
    end

    powerplants -- turn on/off depending of the demand --> temperature
```