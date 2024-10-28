# Modified Reservoir

Based on Reservoir domain, with the following modifications:
- Each reservoir level can vary just between 0 and 100;
- Each release action is restricted between 0 and 30;
- Reservoirs just receive penalties due to overflows

## Small instance

This domain has the following topology, with hexagon nodes representing reservoirs with low amount of rain:

```mermaid
flowchart TD
    t3{{t3}}

    t2 --> t1
    t3 --> t1
    t1 --> Ocean
```

## Medium instance

```mermaid
flowchart TD
    t3{{t3}}
    t6{{t6}}

    t6 --> t3
    t4 --> t2
    t5 --> t2
    t2 --> t1
    t3 --> t1
    t1 --> Ocean
```

## Large instance

```mermaid
flowchart TD
    t3{{t3}}
    t6{{t6}}
    t7{{t7}}
    t8{{t8}}
    t9{{t9}}
    t10{{t10}}
    t11{{t11}}

    t11 --> t7
    t10 --> t7
    t9 --> t6
    t8 --> t6
    t7 --> t3
    t6 --> t3
    t4 --> t2
    t5 --> t2
    t2 --> t1
    t3 --> t1
    t1 --> Ocean
```
