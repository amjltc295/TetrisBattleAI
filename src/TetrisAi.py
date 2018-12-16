from TetrisObject import *
from Util import *
import time

class TetrisAi:
    def __init__( self ):
        pass

    # if there are not valid move, return None
    def getMovementByAi( self, containerOrigin, tetrisBlock ):
        maxScore = -1000000
        movement = BlockMovement( tetrisBlock )
        isValidMoveExsit = False
        for ( aDirection, aFallingBlock ) in enumerate( tetrisBlock.getAllDirectionPos() ):
            for hDelta in range( containerOrigin.getColumnCount() ):
                # aFixBlock has 4 point and can not move but fall down
                aFixBlock = [( row, col + hDelta ) for ( row, col ) in aFallingBlock]
                putState = containerOrigin.getStateOfFallingBlock( aFixBlock )
                if putState.isValid():
                    containerAfterPut = containerOrigin.getCopyContainer()
                    containerAfterPut.putBlockInContainer( putState.getPos() )
                    scoreAtThisPut = self.getScore_BaseAi( containerOrigin, containerAfterPut )
                    isValidMoveExsit = True
                    if scoreAtThisPut > maxScore:
                        movement.setRotationCount( aDirection )
                        movement.setHorizontalDelta( hDelta )
                        movement.setPutPos( putState.getPos() )
                        maxScore = scoreAtThisPut
        if isValidMoveExsit:
            return movement
        else:
            return None

    def getScore_BaseAi( self, containerOrigin, containerAfterPut ):
        score = 0

        # reverse for convenience
        topFilledGrid = [( containerAfterPut.getRowCount() - r ) for r in containerAfterPut.topFilledGridLine]

        # ai by gap
        sortedAbsGap = sorted( getGap( topFilledGrid ) )
        if sortedAbsGap[-2] >= 4 and sortedAbsGap[-1] >= 4:
            score -= 10
            score -= sum( sortedAbsGap[-2:] ) ** 1.1
        score -= getDeviation( topFilledGrid[:-2] ) * 0.35

        # ai by hole
        if containerAfterPut.holeCount >= 1:
            score -= 8
        if containerAfterPut.holeCount >= 2:
            score -= 4
        score -= containerAfterPut.holeCount ** 1.2 * 3

        # ai by blockade
        score -= containerAfterPut.blockadeCount ** 1.5 * 2

        # ai by clear line
        if containerAfterPut.holeCount > 1:
            score += containerAfterPut.lastLineClearCount * 2

        # ai by ready for combo
        if containerAfterPut.holeCount < 2:
            if sum( topFilledGrid[-2:] ) == 0:
                score += 5
            else:
                score -= sum( topFilledGrid[-2:] ) ** 1.3

        # ai by top check
        if max( topFilledGrid ) > 13 and containerAfterPut.holeCount != 0:
            score -= max( topFilledGrid ) * 0.7

        # ai by combo
        if containerAfterPut.filledGridCount > 60:
            score += 10 + containerAfterPut.combo ** 1.5 * 4

        return score

if __name__ == '__main__':
    tetrisContainer = TetrisContainer()
    tetrisContainer.printContainer()
    ai = TetrisAi()

    for i in range( 100 ):
        time.sleep(0.5)
        print(tetrisContainer)
        a = tetrisContainer._area

        print(len(a),len(a[0]))
        print("-------------------------")
        randBlock = TetrisBlock.getRandBlock()
        print(" Got Block: " + randBlock.getBlockName())
        blockMovement = ai.getMovementByAi( tetrisContainer, randBlock )
        print(blockMovement)
        qdw
        if blockMovement != None:
            tetrisContainer.putBlockInContainer( blockMovement.getPutPos() )
            tetrisContainer.printContainer()
            tetrisContainer.printContainerState()
        else:
            print("------ Game Over ------")
            break
