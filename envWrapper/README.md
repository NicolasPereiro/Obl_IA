## "Env Wrapper" para grabar ejecuciones

Para usar la clase env_recorder_wrapper es necesario agregar un par de dependencias al proyecto. 
Puede ejecutar en la terminal:

* poetry add imageio
* poetry add imageio[ffmpeg]

Tras esto, sobreescribir los archivos de environment con su versión compatible a la librería imageio que guarda los videos. 
Agregar el import necesario a la notebook .ipynb, por ejemplo:

from env_recorder_wrapper import VideoRecorderWrapper

Para usarlo, simplemente hay que inicializar el env como siempre y luego inicializar el VideoRecorderWrapper de esta manera:

env = DescentEnv(render_mode="rgb_array")
env_recording = VideoRecorderWrapper(env)

Usando el env_recording en el loop de ejecución obtenemos un video de la ejecución en una carpeta autogenerada
 "/videos".

 Análogamente para TacTix:
env = TacTixEnv(board_size=6, misere=False, render_mode='rgb_array')
env = VideoRecorderWrapper(env)

