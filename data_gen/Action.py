class Action:
  def __init__(self,name,id):
    self.name = name
    self.obj_id = id
  def to_string(self):
    str = "{}({})".format(self.name,self.obj_id)
    #print(str)
    return str