# Referee AI – Arbitraje accesible mediante visión computacional

## Reto

El objetivo de este proyecto es **democratizar el acceso al análisis arbitral en el fútbol**.  
Hoy, herramientas como VAR o tracking deportivo solo están disponibles en ligas profesionales por su alto costo e infraestructura.  
Nuestro reto fue crear una solución que permita que **cualquier persona** pueda analizar una jugada simplemente **subiendo un video**, obteniendo:

- Detección de jugadores, arqueros y árbitros
- Detección y seguimiento del balón
- Mapeo de la jugada sobre el campo
- Trackeo del movimiento a lo largo del tiempo

Todo esto **sin depender de hardware especializado**.

---

## Solución

Desarrollamos un sistema basado en **visión computacional + modelos YOLOv8 nano**, optimizados para correr en dispositivos de bajos recursos.

### Componentes principales

| Funcionalidad | Modelo | Descripción |
|--------------|--------|-------------|
| Detección de jugadores, arqueros y árbitros | YOLOv8 nano (finetuned) | Identifica roles en cada fotograma |
| Detección del balón | YOLOv8 nano (finetuned) | Localiza y sigue el balón en movimiento |
| Detección de puntos de referencia de la cancha | YOLOv8 nano (finetuned) | Permite reconstruir la jugada en coordenadas reales |

### Pipeline

1. El usuario sube un video de una jugada.
2. Los modelos detectan:
   - Jugadores, arqueros y árbitros.
   - Balón.
   - Puntos clave del campo.
3. Se calcula la **transformación de perspectiva** para mapear la jugada sobre una cancha 2D.
4. Se realiza el **trackeo** del balón y/o jugadores.
5. Se genera una visualización final entendible para cualquier persona.

---

## Datos

Entrenamos los tres modelos usando datasets **públicos y abiertos** obtenidos en **Roboflow**, una plataforma colaborativa para etiquetado de imágenes.

Se utilizó **fine-tuning** en cada modelo, ejecutado en múltiples clústeres de cómputo para acelerar el entrenamiento.

---

## ¿Por qué YOLOv8 Nano?

Optamos por modelos **pequeños y livianos** que permiten que la solución:

- Corra en computadoras personales sin GPU
- Se pueda portar a dispositivos móviles
- Sea accesible incluso en contextos con recursos limitados

Esto asegura que la herramienta pueda ser usada por:

- Ligas amateurs
- Escuelas deportivas
- Entrenadores locales
- Aficionados

---

## Próximos pasos

- Determinación automática de faltas y contactos
- Reconocimiento de fuera de juego
- Interfaz web para subir y procesar videos en tiempo real
- App móvil

---


Prototipo desarrollado en el contexto de un hackatón, con enfoque en impacto social y accesibilidad tecnológica.

## 🪪 Licencia

Este proyecto está licenciado bajo la [Creative Commons Attribution 4.0 International License (CC BY 4.0)](https://creativecommons.org/licenses/by/4.0/).

© 2025 Equipo de Desarrollo del Proyecto RaaS — Bytelab, Yachay Tech  
Integrantes: Aldrin Chávez, Ariel Pincay, Britney Robalino, Carlos Castro  

[![License: CC BY 4.0](https://img.shields.io/badge/License-CC--BY%204.0-lightgrey.svg)](https://creativecommons.org/licenses/by/4.0/)


