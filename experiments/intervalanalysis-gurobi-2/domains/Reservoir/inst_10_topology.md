```sh
graph TD
    t1
    t2
    t3([t3])
    t4([t4])
    t5
    t6
    t7([t7])
    t8
    t9
    t10
    ocean

    t1 --> t2
    t1 --> t4
    t1 --> t7
    t2 --> t10
    t2 --> t4
    t2 --> t5
    t2 --> t7
    t9 --> t4
    t9 --> t5
    t9 --> t7
    t9 --> t8
    t9 --> t10
    t8 --> t3
    t4 --> t7
    t6 --> t10
    t6 --> t5
    t6 --> t7
    t5 --> t10
    t5 --> t3
    t5 --> t7
    t7 --> t10
    t7 --> t3
    t3 --> t10
    t10 --> ocean
    ```