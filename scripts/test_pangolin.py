import pangolin
import OpenGL . GL as gl

pangolin . CreateWindowAndBind ("TEST WINDOW", 640 , 480)
gl . glClearColor (0.2 , 0.2 , 0.2 , 1.0)
while not pangolin . ShouldQuit () :
    gl . glClear ( gl . GL_COLOR_BUFFER_BIT | gl . GL_DEPTH_BUFFER_BIT )
    pangolin . FinishFrame ()
