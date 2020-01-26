
            
      (World generation: Done
      (World manipulation: Done 
      (Trajectorie creation: Done
      [extracting features: TODO      later
      [creating aggregations: TODO    later
      [input DC file creation: TODO   Declarative bias type mode
      [Learning of STM: TODO
      {create experiments: TODO 10-fold cross validation IMPORTATNT!
      
      
      Extracting features:
      shape(W,I,Sh)
      posX_t0(W,I,X)
      posY_t0(W,I,X)
      left_of_t0(W,I1,I2)
      right_of_t0(W,I1,I2)
      displX(I,X)
      displY(I,X)
      move_up(W,I,B)
      move_right(W,I,B)
      move_left(W,I,B)
      move_down(W,I,B)
      move_left_of(W,I1,I2,B)
      move_right_of(W,I1,I2,B)
      posX_t1(W,I,X)
      posY_t1(W,I,X)
      left_of_t1(W,I1,I2)
      right_of_t1(W,I1,I2)
      
      need to find closest obj to the left of an obj  # to predict block
      need to find closest obj to the right of an obj #
      need to find rightmost_sphere     #
      need to find leftmost_triangle    # all with aggregations
      need to find rightmost_triangle   #
      need to find leftmost_square      #
      
      if move_left_of(I1,I2) is applied I1 can move I2 can move also other objects that are obstacles can move, how to quanitify displacement? this is reason I was predicting relations here
      
      #
      hier displ: as many objects that move all will have a displ, but super hard to predict, not reversable, but can make search to penalize movements of other objects in hopes that only one object moves: did this
      #
      
      can predict diplacement only in lowest level of the hierarchy
      but i hier level can predict relation, relations can also be predicted in low level of the hierarchy, then we compare their accuracy
      
      but predicting relation in hier much more complicated, many relations change for one relational movement, well also many relations can change we flat. so maybe this is fair
      
      to make scenario more complex: 
        add material as a feature : metal objects move half the distance
        add color as a feature: red objects cant move
                                green objects move only left/right
                                yellow objects move oly up/down
                                
                                
      can have 3 levels in the hierarchy:
        first level: overall, actions: move_left_of or move_right_of
        second level: move_left_of or move_right_of, actions: up/down/left/right
        third level: up/down/left/right, action: displacement
        
        if materials added, only third level needs to be relearned
        if colors added, only second level needs to be relearned
        
        we show state abstraction, easy transfer of knowledge
        
        we need new datasets to learn addition of materials
        we need new dataset to learn addition of colors
        
        changes in the policy: search algorithm
        
      all of this: test with diff number of objects: 3,4,5,3+4+5
                   test with diff noise level in shape and position
                   test with diff number of train examples: maybe?
                    this will show that hier needs less examples
                   
      does not matter what we predict on each level? 
        first level: move_left_of, move_right_of
          relevant features: rels, shape, aggs
        second level: move_up, move_down, move_left, move_down
          relevant features: rels, aggs, color
        third level: displ
          relevant features: pos, aggs, material
          
          
          find reason why performace degrads when increase num of objects
